# PC samplers

Profilers for machines where `perf` is locked (`perf_event_paranoid`), ETW
needs admin rights, or neither exists.  They only record the program counter
at a fixed rate and map it to functions (and lines, with `-g`).

| file | where | how |
|---|---|---|
| `sampler.c` + `report.py` | Linux x86_64 / ppc64le | `LD_PRELOAD` library: SIGPROF on process CPU time every `$SAMPLER_US` µs (default 100); writes `$SAMPLER_OUT.pcs` and `.maps` at exit; `report.py OUT` maps them with addr2line |
| `prof.h` | Linux x86_64, C harnesses | the same in-process: `prof_start()` / `prof_stop("pcs.txt")` around a loop |
| `winprof.py` | Windows x86_64 | a separate process suspends the target's main thread ~1000x/s and reads RIP; COFF symbols (mingw builds) via objdump, `--lines N` via addr2line |
| `case.py` | any | one `tools/compiler_suite.py` benchmark case in a loop, for any of the above |

Linux:

    gcc -O2 -shared -fPIC -o sampler.so tools/sampler/sampler.c -lrt
    SAMPLER_OUT=/tmp/p LD_PRELOAD=$PWD/sampler.so python tools/sampler/case.py lib WAu0012 sparsify:0 10
    python tools/sampler/report.py /tmp/p

Windows (the target must be the base interpreter, not a venv's `python.exe`
launcher, which runs the real one as a child):

    set OBJDUMP=wsl -e x86_64-w64-mingw32-objdump
    python tools\sampler\winprof.py --delay 5 --lines 15 -- C:\Python310\python.exe tools\sampler\case.py lib WAu0012 sparsify:0 10

Inlined code is attributed to the function it was inlined into (e.g. the
zero decoder shows up as `bslz4_driver_run`); `--lines` with a `-g` build
separates it again.
