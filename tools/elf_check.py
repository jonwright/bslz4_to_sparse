#!/usr/bin/env python3
"""
Check a built Linux extension and work out its manylinux wheel tag.

Fails if the .so needs anything but libc/libm/libdl/libpthread (i.e. if
libstdc++ or libgcc_s crept back in), and reports the newest GLIBC symbol
version it references.  That version is the real glibc floor of the binary,
so the wheel tag is derived from it rather than guessed from the build host.

  python3 tools/elf_check.py path/to/_x.so aarch64     # prints manylinux_2_27_aarch64

Uses `readelf`, which reads foreign-architecture ELF files, so it works on a
cross-compiled binary.  Python 2/3 compatible on purpose (old build images).
"""
import re
import subprocess
import sys

ALLOWED = ("libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so",
           "ld-linux", "ld64.so")


def main():
    so, arch = sys.argv[1], sys.argv[2]
    dyn = subprocess.check_output(["readelf", "-d", "-W", so]).decode()
    needed = re.findall(r"NEEDED.*\[([^\]]+)\]", dyn)
    bad = [n for n in needed if not n.startswith(ALLOWED)]
    if bad:
        sys.stderr.write("%s links unexpected libraries: %s\n" % (so, bad))
        sys.exit(1)
    syms = subprocess.check_output(["readelf", "--dyn-syms", "-W", so]).decode()
    vers = [tuple(int(x) for x in v.split("."))
            for v in re.findall(r"GLIBC_(\d+(?:\.\d+)+)", syms)]
    floor = max(vers) if vers else (2, 17)
    # manylinux tags start at 2_17 (the manylinux2014 baseline).
    floor = max(floor[:2], (2, 17))
    sys.stderr.write("%s: needs %s, glibc floor %d.%d\n"
                     % (so, ",".join(needed), floor[0], floor[1]))
    print("manylinux_%d_%d_%s" % (floor[0], floor[1], arch))


if __name__ == "__main__":
    main()
