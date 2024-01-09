.. _installation_and_usage:

Installation and configuration
==============================

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

Configuring the remote repository
---------------------------------

SeisBench uses a remote repository to serve datasets and model weights from.
This repository is hosted on dCache, a high-performance large scale data storage, delivering very high download speeds.
Unfortunately, this comes at the cost of running on a non-standard port for the data transfer that some institutions/providers block.
You can find the server URL by calling `seisbench.remote_root`.
While we highly recommend users to get in touch with their IT department to allow access to our server,
we understand that this might not be possible for all users.
Therefore, we offer a backup server. To use the backup server, simply call `seisbench.use_backup_repository()`.
This will change the repository used within the current runtime.
For a more permanent solution, put the backup repository into your SeisBench config file (usually at `.seisbench/config.json`).
Simply add the following line: ::

    "remote_root": "https://seisbench.gfz-potsdam.de/mirror/"

This will redefine the default, i.e., SeisBench will always access the backup root.
For a more permanent solution, see `the documentation <https://seisbench.readthedocs.io/en/stable/pages/installation_and_usage.html>`_.
