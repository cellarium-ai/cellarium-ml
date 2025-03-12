.. image:: https://cellarium.ai/wp-content/uploads/2024/07/cellarium-logo-medium.png
   :alt: Cellarium Logo
   :width: 180
   :align: center

**Cellarium ML: a machine learning framework for single-cell biology**
======================================================================

Cellarium ML is a PyTorch Lightning-based library for distributed single-cell data analysis.
It provides tools for training deep learning models on large-scale single-cell datasets,
including distributed data loading, model training, and evaluation. Designed to be modular
and extensible, Cellarium ML allows users to easily define custom models, data transformations,
and training pipelines.

-------------------------------------------------------------------------------

**Code Organization**
----------------------

The code is organized as follows:

.. code-block:: text

   cellarium/
   └── ml/
       ├── "callbacks"        # Custom PyTorch Lightning callbacks
       ├── "core"             # Essential components
       │   ├── "CellariumModule"              # PyTorch Lightning Module for model, training step, and optimizer
       │   ├── "CellariumAnnDataDataModule"   # DataModule for multi-GPU DataLoader for AnnData objects
       │   └── "CellariumPipeline"            # Pipeline for data transformations and model inference
       ├── "data"             # Distributed AnnData Collection and multi-GPU Iterable Datasets
       ├── "lr_schedulers"    # Custom learning rate schedulers
       ├── "models"           # Cellarium ML models
       ├── "preprocessing"    # Pre-processing functions
       ├── "transforms"       # Data transformation modules
       ├── "utilities"        # Utility functions for various submodules
       └── "cli.py"           # Implements the "cellarium-ml" CLI. Models must be registered here

Important Notes
~~~~~~~~~~~~~~~

``cellarium/ml/models/*``  
~~~~~~~~~~~~~~~~~~~~~~~~~

- Models must subclass ``CellariumModel`` and implement the following:  
- ``reset_parameters``: Initializes model parameters.  
- ``forward``: Returns a dictionary containing the computed loss under the ``loss`` key.  

Optional hooks for training include:  

- ``on_train_start``: Called at the start of training.  
- ``on_train_epoch_end``: Triggered at the end of each epoch.  
- ``on_train_batch_end``: Triggered at the end of each batch.  

``cellarium/ml/transforms/*``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- All transforms must subclass ``torch.nn.Module``.
- The ``forward`` method must output a dictionary where keys correspond to the input arguments for subsequent transforms or the model.  

``cellarium/ml/cli.py``
~~~~~~~~~~~~~~~~~~~~~~~
- Models must be registered here to be accessible via the command-line interface (``cellarium-ml`` CLI).



-------------------------------------------------------------------------------

**Installation**
-----------------

To install via pip:

.. code-block:: bash

   pip install cellarium-ml

To install the developer version from source:

.. code-block:: bash

   git clone https://github.com/cellarium-ai/cellarium-ml.git
   cd cellarium-ml
   make install  # runs pip install -e .[dev]

**API Documentation and Tutorials**
-----------------------------------

For detailed API documentation and tutorials, visit:  
`Cellarium ML Documentation <https://cellarium-ai.github.io/cellarium-ml/>`_

-------------------------------------------------------------------------------

**For Developers**
-------------------

To run the tests:

.. code-block:: bash

   make test-examples                   # runs single-device cli example tests
   make test-dataloader                 # runs single-device dataloader related tests
   TEST_DEVICES=2 make test-dataloader  # runs multi-device dataloader related test
   make test                            # runs single-device (all other) tests
   TEST_DEVICES=2 make test             # runs multi-device (all other) tests

To format the code automatically:

.. code-block:: bash

   make format                # runs ruff formatter and fixes linter errors

To run the linters:

.. code-block:: bash

   make lint                  # runs ruff linter and checks for formatter errors

To build the documentation:

.. code-block:: bash

   make docs                  # builds the documentation at docs/build/html
