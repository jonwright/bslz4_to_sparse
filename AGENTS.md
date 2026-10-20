# Notes for agents

The single canonical agent guidance for this repository. Merged from the ImageD11
agent notes and this project's own rules. Documentation rules are in
`docs/AGENTS.md`; read that before touching `docs/`, `examples/` or `tools/*docs*`.

## 0. What is the product

The product is the kernels: the CPU decoders and matrix products (`src/*.py`, `src/steps`,
`src/pipeline`) and the GPU ones (`src/cuda`), usable on their own. `src/conveyor` (reading,
workers, sinks, the integrator state, the writers) is a worked example and a verification layer, so
that people see the kernels working at scale. Kernels never import it and nobody is made to use it;
keep it small, optional and honest about being a demo. File formats there are a choice of the demo,
not of the library. The set of kernels will be trimmed at the end; until then, every case stays
tested.

## 1. Think before coding

- Read the files you will touch, end to end. Fragments give confident wrong answers.
- Say what the current code does, and which commit and branch you read, before
  proposing a change. A defect only counts on the branch the work starts from.
- State your assumptions. If there are several readings of the request, list them;
  do not pick one silently. If something is unclear, ask.
- If a simpler approach exists, say so. Push back when warranted.

## 2. Simplicity and surgical changes

- The minimum code that solves the problem. Nothing speculative: no features,
  single-use abstractions or configurability that were not asked for.
- Every changed line must trace to the request. Match the existing style; do not
  "improve" adjacent code. Mention unrelated dead code; remove only what your
  change orphaned.
- Turn the task into checks: a bug -> a test that reproduces it, then passes; a
  refactor -> tests pass before and after.

## 3. Never fudge the numerics

- Never change an algorithm or its numerics to please a test or match an expectation.
- Never invent figures that do not come from the data or the computation.
- Never add workarounds (tolerated slack, special-cased inputs) to make a result
  look better. If a result looks bad, find the cause and fix that.

## 4. Guess or ask

Guess when being wrong costs one edit; ask when it costs a migration.

Yours to decide: names of variables, functions and tests; where a helper lives;
loop structure; equivalent numpy idioms; formatting.

Ask first:

- the name, dtype, layout or meaning of anything written to a file
- the signature or return shape of a function that already has callers
- a default value, where changing it changes results
- the policy for real data: what to drop, merge, mask or pad
- scope: whether a neighbouring problem is in or out

If unsure which list, ask what undoing it would cost. Check the premise of a
question before asking it.

## 5. Measurements and reporting

- Every number carries its conditions: warm cache or cold, synthetic or real, one
  core or many, which CPU. Never write an inference as a measurement. "Not
  measured" is a complete answer.
- Before claiming a change is worth making, ask what would make it worthless, and
  check that first. A saving in one step is often an overhead in the next: measure
  both ends or claim neither.
- Do not test a hypothesis on data generated to fit it. Compare against the
  unchanged code (build both, same machine, same data).
- Report what was checked as checked. Lead with the result and a recommendation.
  A passing, measured result is stated plainly, without hedging.
- Hedge only what is unmeasured, once, naming the missing condition. Do not append
  generic caveats or lists of paths the change does not touch.
- Caution belongs to measurements, contracts and outward actions (push, merge,
  delete). It is not a reason to talk down a verified result.

## 6. Compatibility

Data from the last 25 years must keep working; some is on tape and will come back.

A contract is anything other people's code or files depend on: names, dtypes and
meanings of datasets and attributes; what a public function returns; the output
formats (ImageD11 sparse pixel files, powder files for the ESRF integrator);
defaults that change results. Adding is safe. Renaming, removing, retyping and
redefining are not: new meaning gets a new name. Readers must tolerate attributes
that old files lack. Before changing a contract, list the callers and say what
happens to each. A changed default is a results change: measure before and after,
report both.

`src/*.py` must stay importable on Python 2.7 (CI job `test-py27`) and with numpy
below 2: no infix `@`, f-strings, `yield from`, keyword-only arguments or
`nonlocal`; no `np.random.default_rng`. One binary serves every Python version
(c2py23 resolves the CPython API at runtime), free-threaded builds included. The
subpackages `src/cuda/` and `src/conveyor/` are optional and Python 3 only.

## 7. Build and test

- Submodules (`bitshuffle`, `kcb`, `lz4`, `zstd`) must be checked out:
  `git submodule update --init`.
- Build: `python3 tools/build_extension.py` (puts the `.so` in `src/`, which is
  ignored). Change the c2py spec in `src/bslz4_to_sparse.c`, then regenerate the
  wrapper with `tools/regenerate_wrapper.py`; never edit
  `src/bslz4_to_sparse_wrapper.c` by hand.
- Tests: `BSLZ4_TO_SPARSE_PATH=<dir containing a bslz4_to_sparse package> python3
  -m pytest test`. Needs h5py, hdf5plugin, scipy; some tests and examples also use
  pyFAI.
- `docs/real/` holds example pages run on real data, committed: CI cannot run them.  A change to
  such an example (`PAGES` in `tools/real_data_pages.py`) or to the code it calls is not finished
  until `tools/real_data_pages.py` has been run again and its output committed with it.
- `src/cuda/bslz4_kernels.fatbin` (and `.json`) is built from `src/cuda/*.cu*` and committed: CI has
  no CUDA.  A change to the CUDA source is not finished until the fatbin has been rebuilt with the
  recipe in `tools/build_cuda_fatbin.py` (three toolkits: sm_20, sm_35, the rest) and committed with
  it; `test/test_cuda_fatbin.py` fails otherwise.
- Benchmarks are run by hand on the machine they are for, never in CI and never
  typed into a file (see `docs/AGENTS.md`). Do not write prices or costs into any
  file in the repository; chat only.

## 8. Cluster and GPU runs

- Slurm: always a `low*` partition (`low-gpu`, `low-p9gpu`, `low`), never the normal
  queues; no `--exclusive`. One GPU of one type is enough, no need to sweep all
  cards. `--constraint=a40` etc. picks the type; `sinfo` shows what is idle.
- Python: `module load jupyter-slurm` is the working environment (numpy, scipy, h5py,
  hdf5plugin, pyFAI, pytest, cupy). Do not build new environments.
- p9 (ppc64le, V100, `low-p9gpu`): use the apptainer image as the base. Inside an
  `srun`, the x86 environment leaks in and `module load` is unreliable: run with
  `env -i HOME=<tmp> PATH=/data/id11/jon/p9env/wrappers/p9imaged11:/usr/bin:/bin
  P9IMAGED11_IMAGE=/data/id11/jon/p9env/sif/p9imaged11.sif ...`; `python3` is then
  the image's (3.14, numpy 2, cupy 14). Build the extension in the image with
  `CC=powerpc64le-conda-linux-gnu-gcc CXX=powerpc64le-conda-linux-gnu-g++ python3
  tools/build_extension.py --out <dir>/bslz4_to_sparse` (it needs the `src/*.py`
  copied beside the `.so`). The image also has its own `bslz4_to_sparse`: use
  `PYTHONPATH` as below.
- That environment already has an old `bslz4_to_sparse` (0.0.15) installed.
  `BSLZ4_TO_SPARSE_PATH` is read only by the tests, so anything else (the GPU
  code imports `bslz4_to_sparse._matrix`) silently gets the old one. Put the build
  under test first: `export PYTHONPATH=<dir containing bslz4_to_sparse>`, and print
  `bslz4_to_sparse.__file__` and `.version` at the top of every job log.
- Jobs: `$HOME` is not writable. Set `HOME` to a temp dir and `CUPY_CACHE_DIR`
  under it. Put anything a node must read (builds, data, scripts) on the shared
  home, not in local `/tmp`.
- To iterate, hold an allocation (`salloc --no-shell`, then `srun --jobid=<id>`)
  rather than submitting a batch job per change.

## 9. Design rules for the GPU / stream work

- Threads and forked processes perform alike (measured: 0.93 to 1.07 over eight
  matched pairs; the earlier thread collapse was the decoder's global block
  counters, now off unless a test asks). The conveyor defaults to threads. Allocate
  one big buffer once, before the work starts. No `multiprocessing.shared_memory`.
- No locks of our own on a hot path: work in units of seconds (whole files), and
  synchronise about once per unit, with blocking waits, never polling or spinning.
  Any worker that stops must release whatever its partner waits on.
- Reader, compute and writer are separate components. The compute returns
  indices/values/powder; the output format sits behind an abstract `Sink`.
- Whole-file or large reads for GPFS. O_DIRECT on NFS/GPFS does not help.
- A plain install must need no GPU, cupy or HDF5. Optional parts are optional
  imports.

## 10. Working with others

- Be terse; long text does not get read. Give the conclusion, not what was tried
  and abandoned, in comments, commits and pull requests.
- Stage files by name, one at a time. Never `git add -A` or `git add .`: trees here
  hold large untracked data and test output.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Do not push or open pull requests unless asked; leave merging to a person.
- Never modify a checkout you did not create (`bslz4_to_sparse`, `_llm`, `_dirty`
  are the owner's working copies).
- Everyone here is intelligent and constructive, even when a change arrives in a
  shape that does not fit. Say what is worth keeping and what specifically does not
  work. Disagree with evidence: a case where behaviour differs beats an opinion.
  Resolve issues in the design instead of recording them for later.
