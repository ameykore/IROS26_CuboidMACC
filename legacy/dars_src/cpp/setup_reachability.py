from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os

conda_prefix = os.environ.get("CONDA_PREFIX", "")
boost_include = os.path.join(conda_prefix, "Library", "include")

ext_modules = [
    Pybind11Extension(
        "reachability",
        ["reachability.cpp"],
        include_dirs=[boost_include],
        cxx_std=17,
        extra_compile_args=["/O2"] if sys.platform.startswith("win") else ["-O3"],
    ),
]

setup(
    name="reachability",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
