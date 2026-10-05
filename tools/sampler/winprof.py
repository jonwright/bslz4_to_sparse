"""winprof.py: PC sampler for Windows, no admin rights, no ETW.

  python winprof.py [--delay S] [--top N] [--all-threads] -- command args...

Runs the command and, from this (separate) process, suspends its main thread
about once per millisecond, reads RIP (GetThreadContext) and resumes it.  At
the end the samples are mapped to the modules of the target and, for the
modules holding them, to functions through their COFF symbol table (what
mingw-w64 gcc/clang builds keep unless stripped) with objdump.  Modules
without symbols (MSVC builds, python3xx.dll, ...) are reported by name.

The sampler runs in its own process because a sampler thread inside Python
would wait for the GIL while the extension holds it.

  OBJDUMP   objdump command (default "objdump"); a command starting with
            "wsl" gets /mnt/<drive>/... paths, e.g.
            OBJDUMP="wsl -e x86_64-w64-mingw32-objdump"
  ADDR2LINE addr2line command for --lines (default: OBJDUMP with objdump
            replaced by addr2line); the module needs -g for file:line

Example (one benchmark case, see case.py):

  python tools/sampler/winprof.py --delay 5 -- python tools/sampler/case.py lib WAu0012 sparsify:0 10
"""
import argparse
import bisect
import collections
import ctypes
import os
import shlex
import subprocess
import sys
import time
from ctypes import wintypes

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
winmm = ctypes.WinDLL("winmm")


class THREADENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD)]


class MODULEINFO(ctypes.Structure):
    _fields_ = [("lpBaseOfDll", ctypes.c_void_p), ("SizeOfImage", wintypes.DWORD),
                ("EntryPoint", ctypes.c_void_p)]


H = wintypes.HANDLE
for lib, name, res, args in (
        (k32, "OpenThread", H, (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)),
        (k32, "OpenProcess", H, (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)),
        (k32, "SuspendThread", wintypes.DWORD, (H,)),
        (k32, "ResumeThread", wintypes.DWORD, (H,)),
        (k32, "GetThreadContext", wintypes.BOOL, (H, ctypes.c_void_p)),
        (k32, "CloseHandle", wintypes.BOOL, (H,)),
        (k32, "GetThreadTimes", wintypes.BOOL, (H,) + (ctypes.POINTER(ctypes.c_uint64),) * 4),
        (k32, "CreateToolhelp32Snapshot", H, (wintypes.DWORD, wintypes.DWORD)),
        (k32, "Thread32First", wintypes.BOOL, (H, ctypes.POINTER(THREADENTRY32))),
        (k32, "Thread32Next", wintypes.BOOL, (H, ctypes.POINTER(THREADENTRY32))),
        (psapi, "EnumProcessModulesEx", wintypes.BOOL,
         (H, ctypes.POINTER(H), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD)),
        (psapi, "GetModuleInformation", wintypes.BOOL, (H, H, ctypes.POINTER(MODULEINFO), wintypes.DWORD)),
        (psapi, "GetModuleFileNameExW", wintypes.DWORD, (H, H, wintypes.LPWSTR, wintypes.DWORD))):
    f = getattr(lib, name)
    f.restype, f.argtypes = res, args

THREAD_SUSPEND_RESUME, THREAD_GET_CONTEXT = 0x0002, 0x0008
PROCESS_QUERY_INFORMATION, PROCESS_VM_READ = 0x0400, 0x0010
CONTEXT_CONTROL_AMD64, CONTEXT_SIZE, CONTEXT_FLAGS_OFF, CONTEXT_RIP_OFF = 0x100001, 1232, 0x30, 0xF8


def threads(pid):
    """Thread ids of `pid`, oldest (the main thread) first."""
    snap = k32.CreateToolhelp32Snapshot(0x4, 0)   # TH32CS_SNAPTHREAD
    te = THREADENTRY32()
    te.dwSize = ctypes.sizeof(te)
    out = []
    ok = k32.Thread32First(snap, ctypes.byref(te))
    while ok:
        if te.th32OwnerProcessID == pid:
            out.append(te.th32ThreadID)
        ok = k32.Thread32Next(snap, ctypes.byref(te))
    k32.CloseHandle(snap)

    def created(tid):
        h = k32.OpenThread(0x0800, False, tid)    # THREAD_QUERY_LIMITED_INFORMATION
        t = [ctypes.c_uint64() for _ in range(4)]
        ok = h and k32.GetThreadTimes(h, *map(ctypes.byref, t))
        if h:
            k32.CloseHandle(h)
        return t[0].value if ok else 1 << 64
    return sorted(out, key=created)


def modules(hproc):
    """[(base, size, path)] of the modules loaded in the target."""
    arr = (H * 4096)()
    need = wintypes.DWORD()
    if not psapi.EnumProcessModulesEx(hproc, arr, ctypes.sizeof(arr), ctypes.byref(need), 3):
        return []
    out = []
    for m in arr[:need.value // ctypes.sizeof(H)]:
        mi = MODULEINFO()
        name = ctypes.create_unicode_buffer(1024)
        if psapi.GetModuleInformation(hproc, m, ctypes.byref(mi), ctypes.sizeof(mi)):
            psapi.GetModuleFileNameExW(hproc, m, name, 1024)
            out.append((mi.lpBaseOfDll, mi.SizeOfImage, name.value))
    return sorted(out)


def _tool(env, default):
    cmd = shlex.split(os.environ.get(env, default))
    conv = (lambda p: "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")) if cmd[0] == "wsl" else (lambda p: p)
    return cmd, conv


def lines(path, rvas, image_base):
    """{rva: 'file:line'} through addr2line (needs DWARF, i.e. -g)."""
    default = os.environ.get("OBJDUMP", "objdump").replace("objdump", "addr2line")
    cmd, conv = _tool("ADDR2LINE", default)
    out = subprocess.run(cmd + ["-e", conv(path)] + ["%x" % (image_base + r) for r in rvas],
                         capture_output=True, text=True).stdout.splitlines()
    return {r: os.path.basename(l.split(" (")[0]) for r, l in zip(rvas, out)}


def symbols(path):
    """([(rva, name)] of the functions in a PE file's COFF symbol table, ImageBase)."""
    cmd, conv = _tool("OBJDUMP", "objdump")
    p = conv(path)
    try:
        out = subprocess.run(cmd + ["-p", "-h", "-t", p], capture_output=True, text=True).stdout
    except OSError:
        return [], 0
    # COFF symbol values are offsets into their section: add the section's
    # VMA (from -h; "(sec N)" is section index N-1) and subtract ImageBase
    ib, vma, syms = 0, {}, []
    for line in out.splitlines():
        f = line.split()
        if line.startswith("ImageBase"):
            ib = int(f[1], 16)
        elif len(f) >= 7 and f[0].isdigit() and f[1].startswith("."):
            vma[int(f[0]) + 1] = int(f[3], 16)
        elif "(ty   20)" in line and "(sec " in line:      # functions
            sec = int(line.split("(sec")[1].split(")")[0])
            syms.append((sec, int(f[-2], 16), f[-1]))
    return (sorted((vma[s] + v - ib, n) for s, v, n in syms if s in vma) if ib else []), ib


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait before sampling (imports, setup)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--all-threads", action="store_true", help="sample every thread, not only the main one")
    ap.add_argument("--lines", type=int, default=0, help="also list the N hottest source lines (addr2line)")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    winmm.timeBeginPeriod(1)                       # 1 ms sleeps instead of 15.6 ms
    child = subprocess.Popen(cmd)
    hproc = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, child.pid)
    time.sleep(a.delay)
    buf = ctypes.create_string_buffer(CONTEXT_SIZE + 16)
    ctx = (ctypes.addressof(buf) + 15) & ~15        # CONTEXT is 16-byte aligned
    handles, rips, mods, t_mods = {}, [], [], 0.0
    while child.poll() is None:
        now = time.perf_counter()
        if now - t_mods > 0.5:                       # modules and threads come and go
            mods = modules(hproc) or mods
            tids = threads(child.pid)
            tids = tids if a.all_threads else tids[:1]
            t_mods = now
        for tid in tids:
            h = handles.get(tid) or handles.setdefault(tid, k32.OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, False, tid))
            if not h or k32.SuspendThread(h) == 0xFFFFFFFF:
                continue
            ctypes.c_uint32.from_address(ctx + CONTEXT_FLAGS_OFF).value = CONTEXT_CONTROL_AMD64
            if k32.GetThreadContext(h, ctx):
                rips.append(ctypes.c_uint64.from_address(ctx + CONTEXT_RIP_OFF).value)
            k32.ResumeThread(h)
        time.sleep(0.001)
    winmm.timeEndPeriod(1)

    n = len(rips)
    starts = [m[0] for m in mods]
    bymod = collections.defaultdict(list)
    for r in rips:
        i = bisect.bisect_right(starts, r) - 1
        if i >= 0 and r < mods[i][0] + mods[i][1]:
            bymod[i].append(r - mods[i][0])
        else:
            bymod[-1].append(r)
    funcs = collections.Counter()
    srclines = collections.Counter()
    for i, rvas in bymod.items():
        short = os.path.basename(mods[i][2]) if i >= 0 else "?"
        syms, ib = symbols(mods[i][2]) if i >= 0 and len(rvas) > 0.005 * n else ([], 0)
        if not syms:
            funcs["(%s)" % short] += len(rvas)
            continue
        addrs = [s[0] for s in syms]
        for r in rvas:
            k = bisect.bisect_right(addrs, r) - 1
            funcs["%s (%s)" % (syms[k][1] if k >= 0 else "?", short)] += 1
        if a.lines:
            c = collections.Counter(rvas)
            hot = [r for r, _ in c.most_common(2000)]
            for r, fl in lines(mods[i][2], hot, ib).items():
                k = bisect.bisect_right(addrs, r) - 1
                srclines["%s  [%s]" % (fl, syms[k][1] if k >= 0 else "?")] += c[r]
    print("%d samples" % n)
    for k, v in funcs.most_common(a.top):
        print("  %5.1f%%  %s" % (100.0 * v / max(n, 1), k[:110]))
    if a.lines:
        print("lines:")
        for k, v in srclines.most_common(a.lines):
            print("  %5.1f%%  %s" % (100.0 * v / max(n, 1), k[:130]))


if __name__ == "__main__":
    main()
