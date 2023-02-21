from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(ext_modules=cythonize("e1Numpy.pyx",
                            compiler_directives={'language_level': "3"}
                            ), include_dirs=[numpy.get_include()])


setup(ext_modules=cythonize("gauss_seidel_cython.pyx",
                            compiler_directives={'language_level': "3"}
                            ), include_dirs=[numpy.get_include()])