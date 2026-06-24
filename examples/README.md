# SeisBench examples

This folder contains example notebooks for SeisBench.
For a structured overview of all available examples, check out the main README.

## Writing new examples

If you want to write a new example for SeisBench, please follow these guidelines.
SeisBench examples are grouped into three categories 01 (introduction), 02 (intermediate), and 03 (advanced).
The notebook file should accordingly be called `0Xy_example_name.ipynb` with `X` representing the category and `y` a small letter.
Each example should start with:

- a colab link
- the SeisBench logo
- the command for installing seisbench (`pip install seisbench`)
- a short overview of the content of the tutorial

When adding a tutorial, please also add it to the main README.

All colab links should point to the `main` branch.
Make sure to set the links correctly before merging into the `main` branch.
