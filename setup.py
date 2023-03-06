

"""
Setup script
"""
import setuptools
import os, sys, platform, os.path
from setuptools import setup, Extension
from setuptools.command import build_ext, build_clib
#
import numpy, numpy.f2py   # force wrapper re-generation

if not hasattr( numpy.f2py, 'get_include'):
    numpy.f2py.get_include = lambda : os.path.join(
        os.path.dirname(os.path.abspath(numpy.f2py.__file__)),
        'src')
    
class build_ext_subclass( build_ext.build_ext ):
    def build_extension(self, ext):
        if ext.sources[0].endswith('.pyf'):
            name = ext.sources[0]
            numpy.f2py.run_main( [ name,] )
            ext.sources[0] = os.path.split(name)[-1].replace('.pyf', 'module.c')
            ext.sources.append( os.path.join( numpy.f2py.get_include(), 'fortranobject.c' ) )
        build_ext.build_ext.build_extension(self, ext)

cinc = [ "lz4/lib", "bitshuffle/src" ]
copt = ['-O2', ]

libs = [['bshuf', { 'sources': ["lz4/lib/lz4.c",
                                "bitshuffle/src/bitshuffle_core.c",
                                "bitshuffle/src/iochain.c",],
                     'include_dirs': cinc,
                     'extra_compile_args': copt } ] ] 

ext = Extension( "bslz4_to_sparse",
                 sources = ["src/bslz4_to_sparse.pyf",
                            "src/bslz4_to_sparse.c"],
                 include_dirs  = cinc + [ numpy.get_include(),
                                          numpy.f2py.get_include(), ],
                 extra_compile_args = copt )
    
setup( name = "bslz4_to_sparse" ,
       libraries = libs,
       ext_modules = [ext, ],
       cmdclass = { 'build_ext' : build_ext_subclass },
)
