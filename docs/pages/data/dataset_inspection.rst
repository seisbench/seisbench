.. _dataset_inspection:

Dataset inspection
==================

Dataset inspection utilities are available in the :py:mod:`seisbench.data.inspection` module.
They allow exporting waveform data as MiniSEED, picks and station and event metadata to CSV files for easy visual inspection of the dataset.

To interactively explore the waveforms you can use the Pyrocko Snuffler. Install `Pyrocko <https://pyrocko.org>`_ first:

.. code-block:: bash

    pip install pyrocko PyQt5

Exporting a SeisBench dataset with the DatasetInspection class and inspecting a single event with Pyrocko Snuffler can be done as follows:

.. code-block:: python

    from seisbench.data import DatasetInspection
    import seisbench.data as sbd

    dataset = sbd.ETHZ()
    inspector = DatasetInspection(dataset)
    inspector.export("exported_dataset/")

    # Use Pyrocko Snuffler to inspect a single event
    inspector.pyrocko_snuffle_event(event=42)


The exported directory contains a MiniSEED and metadata dump of the SeisBench dataset.

* ``mseed/``: Contains MiniSEED files, one file per day.
* ``picks/``: Contains pick files in Pyrocko .picks format, one file per day.
* ``events.yaml``: Contains all events in Pyrocko .yaml format.
* ``stations.yaml``: Contains all stations in Pyrocko .yaml format.
* ``all_picks.picks``: Contains dayfile all picks in Pyrocko .picks format

The following command will open the Snuffler with all training data. The events, stations and picks (per day) have to be loaded manually from the exported files.

.. code-block:: bash

    squirrel snuffler -a mseed/

.. figure::  _static/dataset_inspection_snuffler.webp
   :align:   center
