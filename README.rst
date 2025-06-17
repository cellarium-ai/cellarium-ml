# Cellarium ML: Ontology-Aware Cell Type Annotation using scRNA-seq Data

**This branch implements ontology-aware probabilistic models for annotating cell types using single-cell RNA sequencing data. It builds upon the core infrastructure of the Cellarium ML library and introduces a lightweight logistic regression-based model with hierarchical consistency and an ontology-aware evaluation metric.**

---

## 🔬 Project Overview

Current cell type annotation methods often overlook the hierarchical structure of cell types defined in the Cell Ontology. This can lead to inconsistent probabilistic outputs and suboptimal benchmarking. In this work, we address these limitations by introducing:

- **Hierarchical Probability Propagation**: A strategy that ensures predictions are consistent with ontological parent-child relationships.
- **Ontology-Aware Evaluation**: A novel *hop-based F1 scoring* scheme that rewards predictions based on their semantic proximity in the ontology graph.
- **Lightweight Logistic Regression Model**: A scalable model that integrates ontological constraints into a fast, interpretable classification pipeline.

These additions emphasize *annotation over rigid classification* and improve biological interpretability.

---

## 📁 Code Organization

This branch extends the main Cellarium ML repository with the following new components:

- `cellarium/ml/models/custom_logistic_regression`: Contains the modified logistic regression model adapted to handle hierarchical annotation using probability propagation.
- `cellarium/ml/hop_scoring/hop_score_calculation.py`: Implements the hop-based F1 score for ontology-aware benchmarking.
- `cellarium/ml/categorical_distribution/categorical_distribution.py`: Builds over the skeleton base categorical distribution class defined by Pytorch to not perform the normalization step for input propagated probabilities.
- `cellarium/ml/callbacks/prediction_writer.py`: Converts model predictions to hop score outputs and saves df in csv format to the specified GCS/local directory.
- `cellarium/ml/sample_config_files`: Contains YAML configuration files for training and validation runs.
- `cellarium/ml/metadata_files`: Contains important metadata files required to train and validate the custom logistic regression models. The paths to these files are specified in the config files.
- `cellarium/ml/external_benchmarking_details:` Contains mappings for gene names and cell types as well as lists of cell types that are common between the external benchmarking methods and SOCAM. The external benchmarking models include Azimuth, CAS, OnClass and ScTab.

---

## 🚀 Getting Started

To install the package and run this specific model:

```bash
bash
CopyEdit
git clone --single-branch --branch SOCAM https://github.com/cellarium-ai/cellarium-ml.git
cd cellarium-ml
make install

```

To train the logistic regression model with hierarchical constraints:

```bash
bash
CopyEdit
To run model training:
custom_logistic_regression --fit --config SOCAM_train_base_model_config.yaml
To run model inference:
custom_logistic_regression --predict --config SOCAM_test_base_model_config.yaml
```

---

## 📊 Results

We show that:

- The proposed method improves hierarchical consistency of predicted probabilities.
- Ontology-aware F1 score correlates better with expert annotation consensus than traditional F1 metrics.
- The model is scalable to millions of cells and thousands of labels.

---
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
  - Optionally, hooks such as ``.on_train_start``, ``.on_train_epoch_end``, and ``.on_train_batch_end`` can be implemented to be triggered by the ``CellariumModule`` during training phases.
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


