# SeisBench examples

This folder contains example notebooks for SeisBench.
The easiest way of getting started is through our colab notebooks.
Alternatively, you can clone the repository and run the same examples locally.

| Examples                       |  |
|--------------------------------|---|
| Dataset basics                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01a_dataset_basics.ipynb) |
| Model API                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01b_model_api.ipynb) |
| Generator Pipelines            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01c_generator_pipelines.ipynb) |
| Applied picking                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/02a_deploy_model_on_streams_example.ipynb) |
| Using DeepDenoiser             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/02b_deep_denoiser.ipynb) |
| Training PhaseNet (advanced)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03a_training_phasenet.ipynb) |
| Creating a dataset (advanced)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03b_creating_a_dataset.ipynb) |
| Building an event catalog (advanced) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03c_catalog_seisbench_gamma.ipynb) |

## Writing new examples

If you want to write a new example for SeisBench, please follow these guidelines.
SeisBench examples are grouped into three categories 01 (introduction), 02 (intermediate), and 03 (advanced).
The notebook file should accordingly be called `0Xy_example_name.ipynb` with `X` representing the category and `y` a small letter.
Each example should start with:

- a colab link
- the SeisBench logo
- the command for installing seisbench (`pip install seisbench`)
- a short overview of the content of the tutorial

When adding a tutorial, please also add it to:

- the example table at the top of this readme
- the main readme
- the SeisBench documentation examples page

All colab links should point to the `main` branch.
Make sure to set the links correctly before merging into the `main` branch.