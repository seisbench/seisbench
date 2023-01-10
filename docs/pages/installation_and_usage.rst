.. _installation_and_usage:

Installation
======================

Installing via pip
------------------

SeisBench can be installed in two ways.
In both cases, you might consider installing SeisBench in a virtual environment, for example using conda.

SeisBench is available directly through the pip package manager. To install locally run: ::

    pip install seisbench

SeisBench is build on pytorch.
As of pytorch 1.13.0, pytorch is by default shipped with CUDA dependencies which increases the size of the installation considerably.
If you want to install a pure CPU version, the easiest workaround for now is to use: ::

    pip install torch==1.12.1 seisbench

We are working on a `more permanent solution <https://github.com/seisbench/seisbench/issues/141>`_ that allows to use the latest pytorch version in a pure CPU context.

Alternatively, you can install the latest version from source. For this approach, clone `the repository <https://github.com/seisbench/seisbench>`_, switch to the repository root and run: ::

    pip install .

This will install SeisBench in your current python environment from source.
