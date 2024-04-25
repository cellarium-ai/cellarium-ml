*Cellarium ML: distributed single-cell data analysis.*

---------

Cellarium ML is a PyTorch Lightning-based library for distributed single-cell data analysis.
It provides a set of tools for training deep learning models on large-scale single-cell datasets,
including distributed data loading, model training, and evaluation. Cellarium ML is designed to be
modular and extensible, allowing users to easily define custom models, data transformations,
and training pipelines.

Code organization
-----------------

The code is organized as follows:

- ``cellarium/ml/callbacks``: Contains custom PyTorch Lightning callbacks.
- ``cellarium/ml/core``: Includes essential Cellarium ML components:
  - ``CellariumModule``: A PyTorch Lightning Module tasked with defining and configuring the model, training step, and optimizer.
  - ``CellariumAnnDataDataModule``: A PyTorch Lightning DataModule designed for setting up a multi-GPU DataLoader for a collection of AnnData objects.
  - ``CellariumPipeline``: A Module List that pipes the input data through a series of transforms and a model.
- ``cellarium/ml/data``: Contains Distributed AnnData Collection and multi-GPU Iterable Dataset implementations.
- ``cellarium/ml/lr_schedulers``: Contains custom learning rate schedulers.
- ``cellarium/ml/models``: Features Cellarium ML models:
  - Models must subclass ``CellariumModel`` and implement the ``.reset_parameters`` method.
  - The ``.forward`` method should return a dictionary containing the computed loss under the ``loss`` key.
  - Optionally, hooks such as ``.on_train_start``, ``.on_epoch_end``, and ``.on_batch_end`` can be implemented to be triggered by the ``CellariumModule`` during training phases.
- ``cellarium/ml/preprocessing``: Provides pre-processing functions.
- ``cellarium/ml/transforms``: Contains data transformation modules:
  - Each transform is a subclass of ``torch.nn.Module``.
  - The ``.forward`` method should output a dictionary where the keys correspond to the input arguments of subsequent transforms and the model.
- ``cellarium/ml/utilities``: Contains utility functions for various submodules.
- ``cellarium/ml/cli.py``: Implements the ``cellarium-ml`` CLI. Models must be registered here to be accessible via the CLI.

Installation
------------

To install from the pip::

   $ pip install cellarium-ml

To install the developer version from the source::

   $ git clone https://github.com/cellarium-ai/cellarium-ml.git
   $ cd cellarium-ml
   $ make install               # runs pip install -e .[dev]

For developers
--------------

To run the tests::

   $ make test                  # runs single-device tests
   $ TEST_DEVICES=2 make test   # runs multi-device tests

To automatically format the code::

   $ make format               # runs ruff formatter and fixes linter errors

To run the linters::

   $ make lint                  # runs ruff linter and checks for formatter errors

To build the documentation::

   $ make docs                  # builds the documentation at docs/build/html


