

"""
Setup script

use CFLAGS='-narch=native' python3 setup.py to get avx etc
"""
import setuptools
import os, sys, platform, os.path
from setuptools import setup, Extension
from setuptools.command import build_ext, build_clib
#
import numpy, numpy.f2py   # force wrapper re-generation

# patch older numpy:
if not hasattr( numpy.f2py, 'get_include'):
    numpy.f2py.get_include = lambda : os.path.join(
        os.path.dirname(os.path.abspath(numpy.f2py.__file__)),
        'src')

# patch to run f2py during build
class build_ext_subclass( build_ext.build_ext ):
    def build_extension(self, ext):
        if ext.sources[0].endswith('.pyf'):
            name = ext.sources[0]
            numpy.f2py.run_main( [ name,] )
            ext.sources[0] = os.path.split(name)[-1].replace('.pyf', 'module.c')
            ext.sources.append( os.path.join( numpy.f2py.get_include(),
                                              'fortranobject.c' ) )
        build_ext.build_ext.build_extension(self, ext)

if os.path.exists('/proc/cpuinfo'):
    with open('/proc/cpuinfo','r') as fin:
        flags = []
        for line in fin.readlines():
            if line.find('avx2')>=0:
                flags.append('-mavx2')
                break

ext = Extension( "bslz4_to_sparse",
                 sources = ["src/bslz4_to_sparse.pyf",
                            "src/bslz4_to_sparse.c",
                            "src/bshuf.c",
                            "lz4/lib/lz4.c",
                            "bitshuffle/src/bitshuffle_core.c",
                            "bitshuffle/src/iochain.c",  ],
                 include_dirs  = [ numpy.get_include(),
                                   numpy.f2py.get_include(), ],
                 extra_compile_args = flags + [ '-O3',
                                    '-DF2PY_REPORT_ON_ARRAY_COPY=1',
                                        # '-g0', '-flto',
                                        # '-DDEBUG_COPY_ND_ARRAY',
                                        #'-DF2PY_REPORT_ATEXIT'],
                                      ], )

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                       'README.md'), 'r' ) as f:
    readme = f.read()
    
setup( name = "bslz4_to_sparse" ,
       packages = ["bslz4_to_sparse"],
       package_dir = { "bslz4_to_sparse" : "src" },
       ext_package = 'bslz4_to_sparse',
       ext_modules = [ext, ],
       cmdclass = { 'build_ext' : build_ext_subclass },
       install_requires = ["numpy", "h5py"],
       author = 'Jon Wright',
       author_email = 'wright@esrf.fr',
       url = "http://github.com/jonwright/bslz4_to_sparse",
       version = '0.0.6',
       license = 'MIT',
       long_description = readme,
       long_description_content_type='text/markdown',
)
