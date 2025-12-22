.. _speed_up:

Speeding up model application
=============================

When applying models to large datasets, run time is often a major concern.
Here are a few tips to make your model run faster.

 - Run on an accelerator such as a GPU or MPS device. The easiest way to schedule execution on an accelerator is by
   calling ``model.to_preferred_device()`` before applying it. Execution on accelerators is usually faster, even though exact
   speed-ups vary between models. Speed-ups will usually be particularly pronounced for DAS models. However, we note
   that running on GPU is not necessarily the most economic option. For example, in cloud applications it might be
   cheaper (and equally fast) to pay for a handful of CPU machines to annotate a large dataset than for a GPU machine.
 - Use a large ``batch_size``. This parameter can be passed as an optional argument to all models.
   Especially on GPUs, larger batch sizes lead to faster annotations. As long as the batch fits into (GPU) memory,
   it might be worth increasing the batch size.
 - For waveform models, set ``copy=False`` when calling ``annotate``/``classify``. Note that this may lead to in-place
   modifications of the input stream.
 - If you are using torch in version 2.0 or newer, compile your model. It's a simple as running ``model = torch.compile(model)``.
   The compilation will take some time but if you are annotating large amounts of waveforms, it should pay off quickly.
   Note that there are many options for compile that might influence the performance gains considerably.
 - Load data in parallel while executing the model using the asyncio interface, i.e., ``annotate_asyncio`` and ``classify_asyncio``.
   This is usually substantially faster because data loading is IO-bound while the actual annotation is compute-bound.
 - Check if the model is actually your bottleneck or maybe the IO-speed.
 - While SeisBench can automatically resample the waveforms, it can be faster to do the resampling manually beforehand.
   SeisBench uses obspy routines for resampling, which (as of 2023) are not parallelised. Check the required sampling
   rate with ``model.sampling_rate``. Alternative routines are available, e.g., in the Pyrocko library.
 - Any other suggestions or improvements to the SeisBench core? We're always happy to receive pull requests.
