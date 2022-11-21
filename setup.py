import os
from pathlib import Path
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(Path(os.path.dirname(__file__)) / "requirements.txt") as f:
    required = f.readlines()

setup(
    name="seisbench",
    version="0.2.8",
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
    url="https://github.com/seisbench/seisbench",
    packages=find_packages(exclude="tests"),
    python_requires=">=3.7",
    install_requires=required,
)
