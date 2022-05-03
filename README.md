<p align="center">
  <img src="https://raw.githubusercontent.com/seisbench/seisbench/main/docs/_static/seisbench_logo_subtitle_outlined.svg" />
</p>

---

[![PyPI - License](https://img.shields.io/pypi/l/seisbench)](https://github.com/seisbench/seisbench/blob/main/LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/seisbench/seisbench/main_push_action)](https://github.com/seisbench/seisbench)
[![Read the Docs](https://img.shields.io/readthedocs/seisbench)](https://seisbench.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/seisbench)](https://pypi.org/project/seisbench/)
[![Python 3.7](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5568813.svg)](https://doi.org/10.5281/zenodo.5568813)

The Seismology Benchmark collection (*SeisBench*) is an open-source python toolbox for 
machine learning in seismology.
It provides a unified API for accessing seismic datasets and both training and applying machine learning algorithms to seismic data.
SeisBench has been built to reduce the overhead when applying or developing machine learning techniques for seismological tasks.

## Getting started

SeisBench offers three core modules, `data`, `models`, and `generate`.
`data` provides access to benchmark datasets and offers functionality for loading datasets.
`models` offers a collection of machine learning models for seismology.
You can easily create models, load pretrained models or train models on any dataset.
`generate` contains tools for building data generation pipelines.
They bridge the gap between `data` and `models`.

The easiest way of getting started is through our colab notebooks.

| Examples |  |
|---|---|
| Dataset basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01a_dataset_basics.ipynb) |
| Model API | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01b_model_api.ipynb) |
| Generator Pipelines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01c_generator_pipelines.ipynb) |
| Applied picking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/02a_deploy_model_on_streams_example.ipynb) |
| Using DeepDenoiser | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/02b_deep_denoiser.ipynb) |
| Training PhaseNet (advanced) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03a_training_phasenet.ipynb) |
| Creating a dataset (advanced) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03b_creating_a_dataset.ipynb) |

Alternatively, you can clone the repository and run the same [examples](https://github.com/seisbench/seisbench/tree/main/examples) locally.

For more detailed information on Seisbench check out the [SeisBench documentation](https://seisbench.readthedocs.io/).

## Installation

SeisBench can be installed in two ways.
In both cases, you might consider installing SeisBench in a virtual environment, for example using [conda](https://docs.conda.io/en/latest/).

The recommended way is installation through pip.
Simply run:
```
pip install seisbench
```
Alternatively, you can install the latest version from source.
For this approach, clone the repository, switch to the repository root and run:
```
pip install .
```
which will install SeisBench in your current python environment.

## Contributing
There are many ways to contribute to SeisBench and we are always looking forward to your contributions.
Check out the [contribution guidelines](https://github.com/seisbench/seisbench/blob/main/CONTRIBUTING.md) for details on how to contribute.

## Known issues

- Some institutions and internet providers are blocking access to our data and model repository, as it is running on a non-standard port (2443).
  This usually manifests in timeouts when trying to download data or model weights.
  To verify the issue, try accessing [https://dcache-demo.desy.de:2443/](https://dcache-demo.desy.de:2443/) directly from the same machine.
  We are working on a permanent solution for the issue.
  In the meantime, if you are having trouble, try downloading through another network/VPN if possible.
  You can also contact your network administrator to allow access to port 2443 on our server.
  Otherwise, reach out to us, and we will work on finding a solution.

## References
Reference publications for SeisBench:

---

* [SeisBench - A Toolbox for Machine Learning in Seismology](https://doi.org/10.1785/0220210324)

  _Reference publication for software._

---

* [Which picker fits my data? A quantitative evaluation of deep learning based seismic pickers](https://doi.org/10.1029/2021JB023499)

  _Example of in-depth bencharking study of deep learning-based picking routines using the SeisBench framework._

---

