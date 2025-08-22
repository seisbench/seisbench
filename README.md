<p align="center">
  <img src="https://raw.githubusercontent.com/seisbench/seisbench/main/docs/_static/seisbench_logo_subtitle_outlined.svg" />
</p>

---

[![PyPI - License](https://img.shields.io/pypi/l/seisbench)](https://github.com/seisbench/seisbench/blob/main/LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/seisbench/seisbench/main_push.yml?branch=main)](https://github.com/seisbench/seisbench)
[![Read the Docs](https://img.shields.io/readthedocs/seisbench)](https://seisbench.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/seisbench)](https://pypi.org/project/seisbench/)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
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

| Examples                                         |                                                                                                                                                                                                         |
|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dataset basics                                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01a_dataset_basics.ipynb)                  |
| Model API                                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01b_model_api.ipynb)                       |
| Generator Pipelines                              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01c_generator_pipelines.ipynb)             |
| Applied picking                                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/02a_deploy_model_on_streams_example.ipynb) |
| Using DeepDenoiser                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/02b_deep_denoiser.ipynb)                   |
| Depth phases and earthquake depth                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/02c_depth_phases.ipynb)                    |
| Training PhaseNet (advanced)                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03a_training_phasenet.ipynb)               |
| Creating a dataset (advanced)                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03b_creating_a_dataset.ipynb)              |
| Training Denoiser (advanced)                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03e_training_denoiser.ipynb)               |
| Building an event catalog with GaMMA (advanced)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03c_catalog_seisbench_gamma.ipynb)         |
| Building an event catalog with PyOcto (advanced) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03d_catalog_seisbench_pyocto.ipynb)        |

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

### CPU only installation

SeisBench is built on pytorch, which in turn runs on CUDA for GPU acceleration.
Sometimes, it might be preferable to install pytorch without CUDA, for example, because CUDA will not be used and the CUDA binaries are rather large.
To install such a pure CPU version, the easiest way is to follow a two-step installation.
First, install pytorch in a pure CPU version [as explained here](https://pytorch.org/).
Second, install SeisBench the regular way through pip.
Example instructions would be:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install seisbench
```

## Contributing
There are many ways to contribute to SeisBench and we are always looking forward to your contributions.
Check out the [contribution guidelines](https://github.com/seisbench/seisbench/blob/main/CONTRIBUTING.md) for details on how to contribute.

## Known issues

- Some institutions and internet providers are blocking access to our data and model repository, as it is running on a non-standard port (2880).
  This usually manifests in timeouts when trying to download data or model weights.
  To verify the issue, try accessing [https://hifis-storage.desy.de:2880/](https://hifis-storage.desy.de:2880/) directly from the same machine.
  As a mitigation, you can use our backup repository. Just run `seisbench.use_backup_repository()`.
  Please note that the backup repository will usually show lower download speeds.
  We recommend contacting your network administrator to allow outgoing access to TCP port 2880 on our server as a higher performance solution.
- We've recently changed the URL of the SeisBench repository. To use the new URL update to SeisBench 0.4.1.
  It this is not possible, you can use the following commands within your runtime to update the URL manually:
  ```python
  import seisbench
  from urllib.parse import urljoin

  seisbench.remote_root = "https://hifis-storage.desy.de:2880/Helmholtz/HelmholtzAI/SeisBench/"
  seisbench.remote_data_root = urljoin(seisbench.remote_root, "datasets/")
  seisbench.remote_model_root = urljoin(seisbench.remote_root, "models/v3/")
  ```
- On the Apple M1 and M2 chips, pytorch seems to not always work when installed directly within `pip install seisbench`.
  As a workaround, follow the instructions at (https://pytorch.org/) to install pytorch and then install SeisBench as usual through pip.
- EQTransformer model weights "original" in version 1 and 2 are incompatible with SeisBench >=0.2.3. Simply use `from_pretrained("original", version="3")` or `from_pretrained("original", update=True)`. The weights will not differ in their predictions.

## References
Reference publications for SeisBench:

---

* [SeisBench - A Toolbox for Machine Learning in Seismology](https://doi.org/10.1785/0220210324)

  _Reference publication for software._

---

* [Which picker fits my data? A quantitative evaluation of deep learning based seismic pickers](https://doi.org/10.1029/2021JB023499)

  _Example of in-depth bencharking study of deep learning-based picking routines using the SeisBench framework._

---

## Acknowledgement

The initial version of SeisBench has been developed at [GFZ Potsdam](https://www.gfz-potsdam.de/) and [KIT](https://www.gpi.kit.edu/) with funding from [Helmholtz AI](https://www.helmholtz.ai/).
The SeisBench repository is hosted by [HIFIS - Helmholtz Federated IT Services](https://www.hifis.net/).
