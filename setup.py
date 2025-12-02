import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "seisbench.ext.stack_windows",
            sources=["seisbench/ext/stack_windows.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-flto", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
    ]
)
