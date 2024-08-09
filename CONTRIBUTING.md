<p align="center">
  <img src="https://raw.githubusercontent.com/seisbench/seisbench/main/docs/_static/seisbench_logo_subtitle_outlined.svg" />
</p>

---

# Contributing to SeisBench

SeisBench is an open-source project and encourages community engagement.
This document gives a short overview of ways to contribute to SeisBench.
There are multiple ways in which you can contribute:

- *Report a bug:* To report a suspected bug, please raise an issue on Github. Please try to give a short but complete description of the issue.
- *Suggest a feature:* To suggest a new feature, please raise an issue on Github. Please describe the feature and the intended use case.
- *Adding a dataset or model:* We are always happy to extend SeisBench with new datasets or models.
  If you would like to contribute a dataset or model, please open a pull request following the steps outlined below.
  As a general rule of thumb regarding whether we will add a dataset or model, we typically require a peer-reviewed publication accompanying the dataset or model.
  However, this is not fixed, and we might deviate from the rule on a case-by-case basis. Feel free to raise an issue if unsure.

  Note that you can also implement models and datasets using SeisBench without including them in the SeisBench repository.
  To this end, simply include `seisbench.data.BenchmarkDataset` or `seisbench.models.WaveformModel` and make your class inherit from it.
  Doing so will also ease the process of adding a dataset or model to SeisBench.

  SeisBench makes some dataset and pretrained model weights available through its data repository.
  If you are contributing a new dataset or model and would like to make it available through the SeisBench data repository,
  please raise an issue or mention this in the pull request.
  We'll then discuss possibilities.

When contributing, please include the appropriate label with the issue submission on GitHub where possible:

- Bug --> **'bug'**
- Feature request --> **'enhancement'**
- For dataset or model addition
  - dataset addition --> **'dataset extension'**
  - model addition --> **'model extension'**

## Installing SeisBench for development purposes

To develop SeisBench code, you'll need a development install of SeisBench.
You can install a development version of SeisBench using the following commands (assuming conda is installed):
```bash
conda create --name seisbench python=3.9
conda activate seisbench
git clone git@github.com:seisbench/seisbench.git
cd seisbench
pip install -e .
```

For development, we further strongly recommend installing `pytest` and the pre-commit hook for the [Black](https://black.readthedocs.io/en/stable/) formatter.
This can be done by running the following commands in the root directory.
```bash
pip install pytest
pip install pre-commit
pre-commit install
```

For running the tests use:
```bash
pytest tests/
```

For building the documentation, run the following commands:
```bash
cd docs
pip install -r requirements.txt
make html
```
The output will appear in `docs/_build/html`.

## Building the documentation

In case your change touches the documentation, you might want to build the documentation.
To build the documentation first install the requirements from `docs/requirements.txt`.
Then run `make html` to build the docs in html format.
The output will appear in `docs/_build/html`.

## Code structure

SeisBench consists of:
- the `seisbench` module
- three main submodules: `seisbench.data`, `seisbench.models` and `seisbench.generate`
- the auxiliary submodule `seisbench.util`

The module `seisbench` contains global configurations, e.g., cache root, remote root and logging.
All submodules may import the `seisbench` module.

The submodule `seisbench.data` contains everything connected to datasets.
This includes the base classes, the benchmark datasets and tools for creating datasets.
The submodule `seisbench.models` contains everything connected to models.
This includes the base classes and the different models.
The submodule `seisbench.generate` contains everything for creating training pipelines.
This includes the generators as well as a collection of augmentations.
The three submodules have a clear import structure among each other to avoid circular dependencies.
Only `seisbench.generate` may import `seisbench.data`, while no other imports between the modules
are used.

The submodule `seisbench.util` is a loose collection of auxiliary functions, e.g., for file access.
This module can be imported from any other module.

In general, all functionality in SeisBench should be directly accessible from the submodules.
This means imports should always be, e.g., `import seisbench.model` and not `import seisbench.model.xyz`.
To this end, all publicly exposed functionality needs to be imported in the respective `__init__.py` files.

## Versioning and branching model

SeisBench uses [semantic versioning](https://semver.org/), i.e., version numbers have the form X.Y.Z, with X the major version, Y the minor version, and Z the patch version.
SeisBench uses a simple branching model, inspired by the [model used in obspy](https://github.com/obspy/obspy/wiki/ObsPy-Git-Branching-Model).
There are two types of branches, the main branch and the maintenance branches.
The main branch contains both new features and bugfixes.
New minor and major versions always result from the main branch.
In contrast, maintenance branches only contain bugfixes and do **not** contain new features.
Bugfixes are merged from the maintenance branches into the main branch.

SeisBench follows the [numpy schedule](https://numpy.org/neps/nep-0029-deprecation_policy.html) for supported python versions.
This does not necessarily mean that new SeisBench versions will not run on older python versions, but that we will not ship wheels for these versions.

## Submitting a pull request

You want to contribute to SeisBench? That's great news. Here's a quick guide.

1. Fork the repository
1. Make a new branch. If your contribution is a new feature, base your branch on `main`.
   For bugfixes, choose the correct `maintenance_*` branch as base.
   If unsure, check the branching policy above or open an issue to discuss.
1. Write your code.
1. Add tests to your code. Any new feature or bugfix needs a test.
1. Make sure the code style is correct. SeisBench uses the [black code formatter](https://github.com/psf/black). Enforcing code style can easily be achieved by installing the pre-commit hook mentioned above.
1. When adding a new feature, make sure the feature is properly documented.
1. Push to your fork and open a pull request. We will review the PR and may suggest some changes or improvements.

**SeisBench is distributed under GPLv3 license.
Please ensure that all the code you contribute can be distributed under this license.
Sending a PR implies that you agree with this.**
