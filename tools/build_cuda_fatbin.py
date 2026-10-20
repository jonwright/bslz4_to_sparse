#!/usr/bin/env python3
"""Compile the CUDA kernels (src/cuda/*.cu*) ahead of time into one fatbin, for the driver-API
backend (src/cuda/_backend_drv.py): no NVRTC, cupy or CUDA runtime at run time.

    python3 tools/build_cuda_fatbin.py --nvrtc .../libnvrtc.so.12 --fatbinary .../bin/fatbinary \
        --out DIR

writes DIR/bslz4_kernels.fatbin (a cubin for every architecture in --arch, and PTX for the last,
which the driver compiles for newer GPUs) and DIR/bslz4_kernels.json (kernel name as _gpu.py asks
for it, e.g. "count_above<unsigned short>", -> mangled symbol).  Every template instantiation
_gpu.py can ask for is built: each pixel dtype, accumulator and frames-per-read.

The compiler is NVRTC with the options cupy gives it, so the machine code is the one the cupy
backend runs (nvcc's front end gives different PTX for the same source, and ptxas a different
schedule: 1.5% slower decoding measured on a GTX TITAN X).  libnvrtc: pip nvidia-cuda-nvrtc-cu12;
fatbinary: the CUDA toolkit's (redist cuda_nvcc).  CUDA 12.x is the last toolkit for compute
capability below 7.5 (Maxwell, Pascal, Volta).

--nvrtc-for 35=.../cuda-11.8/lib64/libnvrtc.so --nvrtc-for 20=.../cuda-8.0/lib64/libnvrtc.so adds
cubins, each from the toolkit given, for GPUs CUDA 12 dropped: Kepler (CUDA 11.8 is the last for
it) and Fermi (CUDA 8.0; an sm_20 cubin also runs on CC 2.1).  Before CUDA 11.1 NVRTC gives PTX
only: the toolkit's bin/ptxas makes the cubin.  Driver 340 (CUDA 6.5) loads such cubins.  Each
NVRTC loads its libnvrtc-builtins from LD_LIBRARY_PATH: list every toolkit's lib64, CUDA 8's first
(it asks for the unversioned libnvrtc-builtins.so).

The shipped src/cuda/bslz4_kernels.fatbin (CI has no CUDA; test/test_cuda_fatbin.py checks it
matches the source), at the ESRF (the cuda modules conflict, so 11.8 is named by its path; CUDA 8
is not on cvmfs: unpacked from NVIDIA's cuda_8.0.61_375.26_linux.run, md5 33e1bd98...):

    module load cuda/12.6.2
    C118=/cvmfs/hpc.esrf.fr/software/packages/ubuntu24.04/x86_64/cuda/11.8.0
    C80=/data/id11/jon/testcrunchGPU/toolkits/cuda-8.0
    LD_LIBRARY_PATH=$C80/lib64:$C118/lib64:$CUDA_HOME/lib64 python3 tools/build_cuda_fatbin.py \
        --nvrtc $CUDA_HOME/lib64/libnvrtc.so.12 --fatbinary $CUDA_HOME/bin/fatbinary \
        --nvrtc-for 35=$C118/lib64/libnvrtc.so --nvrtc-for 20=$C80/lib64/libnvrtc.so --out src/cuda"""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "cuda")
CTYPES = ["unsigned char", "signed char", "unsigned short", "short", "unsigned int", "int", "float"]
ACCS = [("float", "float"), ("float", "double"), ("unsigned int", "long long")]  # f32, f64, i64
FRAMES = [1, 2]  # _gpu.FRAMES_PER_READ and 1
FILES = ["nvcomp_lz4.cuh", "shuffle_warp.cuh", "bslz4_split.cu"]  # concatenated as by _gpu.py
PLAIN = ["unblock", "decompress", "scan_counts", "choose_level", "scan_images"]  # extern "C"
ARCHS = "50,52,60,61,70,75,80,86,89,90"
OPTIONS = ["--std=c++17", "-ftz=true", "--device-as-default-execution-space"]  # as cupy


def kernel_names():
    names = []
    for t in CTYPES:
        names += ["count_above<%s>" % t, "collect_write<%s>" % t]
        names += ["csr1d<%s, %s, %s, %d>" % (t, w, a, f) for w, a in ACCS for f in FRAMES]
    return names


class Nvrtc(object):
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        self.ptxas = os.path.join(os.path.dirname(os.path.abspath(path)), "..", "bin", "ptxas")
        ma, mi = ctypes.c_int(), ctypes.c_int()
        self.lib.nvrtcVersion(ctypes.byref(ma), ctypes.byref(mi))
        self.version = (ma.value, mi.value)
        self.options = OPTIONS if ma.value >= 11 else [  # as cupy, in what an old NVRTC accepts
            "--std=c++14" if ma.value >= 9 else "--std=c++11", "-ftz=true", "-default-device"]

    def check(self, rc, prog=None):
        if rc:
            self.lib.nvrtcGetErrorString.restype = ctypes.c_char_p
            msg = self.lib.nvrtcGetErrorString(rc).decode()
            if prog is not None:
                n = ctypes.c_size_t()
                self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
                log = ctypes.create_string_buffer(n.value)
                self.lib.nvrtcGetProgramLog(prog, log)
                msg += "\n" + log.value.decode()
            raise SystemExit("NVRTC: " + msg)

    def compile(self, src, names, arch):
        """(cubin bytes for sm_<arch> or PTX for compute_<arch>, {name: lowered name})"""
        if arch.startswith("sm_") and not hasattr(self.lib, "nvrtcGetCUBIN"):
            ptx, lowered = self.compile(src, names, "compute_" + arch[3:])
            with tempfile.TemporaryDirectory() as tmp:
                p, c = os.path.join(tmp, "k.ptx"), os.path.join(tmp, "k.cubin")
                with open(p, "wb") as fh:
                    fh.write(ptx.rstrip(b"\0"))
                subprocess.check_call([self.ptxas, "-arch=" + arch, p, "-o", c])
                return open(c, "rb").read(), lowered
        L, prog = self.lib, ctypes.c_void_p()
        self.check(L.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"bslz4_kernels.cu", 0,
                                        None, None))
        for n in names:
            self.check(L.nvrtcAddNameExpression(prog, n.encode()))
        opts = [o.encode() for o in self.options + ["-arch=%s" % arch]]
        self.check(L.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)),
                   prog)
        lowered = {}
        for n in names:
            p = ctypes.c_char_p()
            self.check(L.nvrtcGetLoweredName(prog, n.encode(), ctypes.byref(p)))
            lowered[n] = p.value.decode()
        kind = "CUBIN" if arch.startswith("sm_") else "PTX"
        size = ctypes.c_size_t()
        self.check(getattr(L, "nvrtcGet%sSize" % kind)(prog, ctypes.byref(size)))
        buf = ctypes.create_string_buffer(size.value)
        self.check(getattr(L, "nvrtcGet%s" % kind)(prog, buf))
        L.nvrtcDestroyProgram(ctypes.byref(prog))
        return buf.raw, lowered


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nvrtc", default="libnvrtc.so.12")
    ap.add_argument("--fatbinary", default="fatbinary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default=ARCHS, help="compute capabilities, e.g. 50,52,61")
    ap.add_argument("--nvrtc-for", action="append", default=[], metavar="ARCH=LIBNVRTC",
                    help="an extra cubin from another toolkit, e.g. 35=.../libnvrtc.so.11.2")
    a = ap.parse_args()
    nv = Nvrtc(a.nvrtc)
    other = dict(x.split("=", 1) for x in a.nvrtc_for)
    old = sorted(other, key=int)
    nvs = {c: Nvrtc(other[c]) for c in old}
    names = kernel_names()
    src = "".join(open(os.path.join(SRC, f)).read() for f in FILES)
    archs = a.arch.split(",")
    os.makedirs(a.out, exist_ok=True)
    fatbin = os.path.join(a.out, "bslz4_kernels.fatbin")
    with tempfile.TemporaryDirectory() as tmp:
        images, table = [], None
        for c in old + archs + [None]:
            arch = "sm_%s" % c if c else "compute_%s" % archs[-1]
            data, lowered = nvs.get(c, nv).compile(src, names, arch)
            if table is not None and lowered != table:
                raise SystemExit("lowered names differ between architectures")
            table = lowered
            path = os.path.join(tmp, arch + (".cubin" if c else ".ptx"))
            with open(path, "wb") as fh:
                fh.write(data)
            images.append("--image3=kind=%s,sm=%s,file=%s" % ("elf" if c else "ptx",
                                                              c or archs[-1], path))
        subprocess.check_call([a.fatbinary, "--create=" + fatbin, "-64"] + images)
    table.update({n: n for n in PLAIN})
    built = {c: "%d.%d" % nvs.get(c, nv).version for c in old + archs}  # test_cuda_fatbin.py
    table["_build"] = {"files": FILES, "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
                       "nvrtc": built}
    with open(os.path.join(a.out, "bslz4_kernels.json"), "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote %s (%.1f MB, %d kernels, sm_%s + compute_%s PTX)"
          % (fatbin, os.path.getsize(fatbin) / 1e6, len(table) - 1, ",".join(old + archs),
             archs[-1]))


if __name__ == "__main__":
    sys.exit(main())
