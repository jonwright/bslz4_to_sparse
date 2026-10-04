#!/bin/sh
# Run tools/compiler_suite.py with the jupyter-slurm python built for THIS
# machine's architecture, taken straight from cvmfs (no module load needed:
# on p9 the default module is an old ubuntu20.04 one, and a PATH inherited
# from an x86_64 session would pick the x86_64 python).
#
#   sh /home/esrf/wright/git/bslz4_to_sparse_llm/tools/compiler_suite.sh all
#
# JS_VERSION picks another jupyter-slurm version; BSLZ4_SUITE_PYTHON any python.
arch=$(uname -m)
case $arch in
  x86_64)  ver=${JS_VERSION:-2025.04.6} ;;
  ppc64le) ver=${JS_VERSION:-2023.10.11} ;;
  *)       ver=${JS_VERSION:-latest} ;;
esac
env=/cvmfs/hpc.esrf.fr/software/packages/linux/$arch/jupyter-slurm/$ver/envs/jupyter-slurm
py=${BSLZ4_SUITE_PYTHON:-$env/bin/python3}
if [ ! -x "$py" ]; then
  echo "compiler_suite.sh: no python at $py (set JS_VERSION or BSLZ4_SUITE_PYTHON)" >&2
  exit 1
fi
# that env's gcc/g++ first; nothing from another architecture's python
PATH=$(dirname "$py"):$PATH
export PATH
unset PYTHONHOME PYTHONPATH
exec "$py" "$(dirname "$0")/compiler_suite.py" "$@"
