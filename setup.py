from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="seisbench",
    version="0.1.0",
    author="Jack Woollam,Jannes Münchmeyer",
    author_email="jack.woollam@kit.edu,munchmej@gfz-potsdam.de",
    description="The seismological machine learning benchmark collection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seisbench/seisbench",
    packages=find_packages(exclude="tests"),
    python_requires=">=3.6",
    install_requires=required,
)
