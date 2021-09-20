import os
import subprocess
import warnings

from pathlib import Path
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(Path(os.path.dirname(__file__)) / "requirements" / "req_dependent.txt") as fd:
    dependant_req = fd.readlines()

with open(Path(os.path.dirname(__file__)) / "requirements" / "req_common.txt") as fc:
    common_req = fc.readlines()


def exit_code_callback(process):
    _stdout, _stderr = process.communicate()

    if process.returncode != 0:
        warnings.warn(
            f"Command '{' '.join(process.args)}' exited with code {process.returncode}."
        )


if os.name == "nt":
    # Windows: Obtain windows compatible python pkg binaries via pipwin
    for pkg in ("wheel", "pipwin"):
        process = subprocess.Popen(
            ["pip", "install", pkg], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        exit_code_callback(process)
    for pkg in dependant_req:
        process = subprocess.Popen(
            ["pipwin", "install", pkg], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        exit_code_callback(process)
    required = common_req
else:
    # Linux, Mac OS: Installation via standard pip - GDAL not req.
    required = common_req + dependant_req[2:4]

setup(
    name="seisbench",
    version="0.1.3",
    author="Jack Woollam,Jannes Münchmeyer",
    author_email="jack.woollam@kit.edu,munchmej@gfz-potsdam.de",
    description="The seismological machine learning benchmark collection",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPLv3",
    url="https://github.com/seisbench/seisbench",
    packages=find_packages(exclude="tests"),
    python_requires=">=3.6",
    install_requires=required,
)
