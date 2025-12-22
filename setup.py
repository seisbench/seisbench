import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "seisbench.ext.utils",
            sources=["seisbench/ext/utils.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-flto"],
            extra_link_args=[],
        ),
    ]
)
