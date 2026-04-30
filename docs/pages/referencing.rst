Referencing Guide
=================

SeisBench is a collaborative project that builds on contributions from many
researchers. When using SeisBench, you typically rely on work from different
authors. This guide outlines which publications to cite for common use cases.

Applying a Pretrained Model
---------------------------

If you are applying a pretrained model, please cite:

- SeisBench
- The model architecture (see ``model.citation`` attribute)
- The publication describing the trained weights (see ``model.weights_docstring`` attribute).
- The publication describing the training dataset

.. admonition:: Example statement

   "We use PhaseNet [1] implemented in SeisBench [2] trained on the INSTANCE dataset [3, 4]."

Using a Dataset
---------------

If you are using a dataset provided through SeisBench, please cite:

- SeisBench
- The dataset publication

.. admonition:: Example statement

   "We use the INSTANCE dataset [1] accessed through SeisBench [2]."

Training a Model
----------------

If you are training your own model, please cite:

- SeisBench
- The model architecture
- The training dataset

.. admonition:: Example statement

   "We train the PhaseNet model [1], implemented in SeisBench [2], on the INSTANCE dataset [3]."

Closing Remarks
---------------

Proper citation is essential to give credit to the original authors and to
support continued development of methods and software. In many cases, you
should also consider citing ObsPy, as it is underlies SeisBench and is
commonly used alongside SeisBench.
