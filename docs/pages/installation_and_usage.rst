.. _installation_and_usage:

Installation and Configuration
==============================

Installing via pip
------------------

SeisBench can be installed in two ways.
In both cases, you might consider installing SeisBench in a virtual environment, for example using conda.

SeisBench is available directly through the pip package manager. To install locally run: ::

    pip install seisbench

By default, SeisBench will be installed without support for DAS data to avoid a range of additional dependencies.
To install these, you can use: ::

    pip install seisbench[das]

If you want, you can install the latest version from source. For this approach, clone `the repository <https://github.com/seisbench/seisbench>`_, switch to the repository root and run: ::

    pip install .

This will install SeisBench in your current python environment from source.
As before, you can explicity specify that the DAS dependencies should be installed too by appending ``[das]``.

CPU only installation
^^^^^^^^^^^^^^^^^^^^^

SeisBench is built on pytorch, which in turn runs on CUDA for GPU acceleration.
Sometimes, it might be preferable to install pytorch without CUDA, for example, because CUDA will not be used and the CUDA binaries are rather large.
To install such a pure CPU version, the easiest way is to follow a two-step installation.
First, install pytorch in a pure CPU version `as explained here <https://pytorch.org/>`_.
Second, install SeisBench the regular way through pip.
Example instructions would be: ::

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install seisbench

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

    "remote_root": "https://seisbench.gfz.de/mirror/"

This will redefine the default, i.e., SeisBench will always access the backup root.
