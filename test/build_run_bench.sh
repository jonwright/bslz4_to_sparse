#!/bin/bash



git clone https://github.com/jonwright/bslz4_to_sparse $HOSTNAME
cd $HOSTNAME
cp /proc/cpuinfo .

git submodule init
git submodule update

python3 -m venv ./venv_std
source ./venv_std/bin/activate
CFLAGS='-march=native' python -m pip install .
cd test
python -m pip install hdf5plugin
python bench1.py | tee bench1_std.out
cd ..
deactivate


python3 -m venv ./venv_kcb
source ./venv_kcb/bin/activate
CFLAGS='-march=native' USE_KCB=1 python -m pip install .
cd test
python -m pip install hdf5plugin
python bench1.py | tee bench1_kcb.out
cd ..
deactivate











