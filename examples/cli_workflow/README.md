# Example workflow

This example workflow demonstrates how to:

1. Compute the mean and standard deviation of normalized and log1p-transformed data.
2. Perform PCA on data that has been z-scored based on the mean and standard deviation from step 1.
3. Train a logistic regression classifier on the principal component embeddings of the data.

To execute the workflow, run the scripts in sequence:

```bash
cellarium-ml onepass_mean_var_std fit --config onepass_train_config.yaml
cellarium-ml incremental_pca fit --config ipca_train_config.yaml
cellarium-ml logistic_regression fit --config lr_train_config.yaml
```

## `OnePassMeanVarStd`

Generate a default config file:

```
cellarium-ml onepass_mean_var_std fit --print_config > onepass_train_config.yaml
```

Below we highlight the changes made to the default configuration file.

### trainer

Change the number of devices and set the path for logs and weights:

```diff
<   devices: auto
<   default_root_dir: null
---
>   devices: 2
>   default_root_dir: /tmp/test_examples/onepass
```

### model

Add `NormalizeTotal` and `Log1p` transforms:

```diff
<   transforms: null
---
>   transforms:
>     - class_path: cellarium.ml.transforms.NormalizeTotal
>       init_args:
>         target_count: 10_000
>     - cellarium.ml.transforms.Log1p
```

Change `OnePassMeanVarStd`'s algorithm to `shifted_data`:

```diff
<       algorithm: naive
---
>       algorithm: shifted_data
```

### data

Configure the `DistributedAnnDataCollection`. Here we validate `obs` columns that are used by the transforms and the model (`total_mrna_umis`). Validation is done for each loaded AnnData file against the first (reference) AnnData file by checking that column names and dtypes match between the two:

```diff
<       filenames: null
<       shard_size: null
<       max_cache_size: 1
<       obs_columns_to_validate: null
---
>       filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad
>       shard_size: 100
>       max_cache_size: 2
>       obs_columns_to_validate:
>         - total_mrna_umis
```

Configure the `DataLoader`. `batch_keys` have to include all input arguments to the transforms and the model. For example, `NormalizeTotal`'s arguments are `x_ng` and `total_mrna_umis_n`, `Log1p`'s argument is `x_ng`, and `OnePassMeanVarStd`'s arguments are `x_ng` and `var_names_g`:

```diff
<   batch_keys: null
<   batch_size: 1
<   num_workers: 0
---
>   batch_keys:
>     x_ng:
>       attr: X
>       convert_fn: cellarium.ml.utilities.data.densify
>     var_names_g:
>       attr: var_names
>     total_mrna_umis_n:
>       attr: obs
>       key: total_mrna_umis
>   batch_size: 100
>   num_workers: 2
```

## `IncrementalPCA`

Generate a default config file:

```
cellarium-ml incremental_pca fit --print_config > ipca_train_config.yaml
```

Below we highlight the changes made to the default configuration file.

### train

Change the number of devices and set the path for logs and weights:

```diff
<   devices: auto
<   default_root_dir: null
---
>   devices: 2
>   default_root_dir: /tmp/test_examples/ipca
```

### model

Add `NormalizeTotal` and `Log1p`, and `ZScore` transforms. Note, that `mean_g`, `std_g`, and `var_names_g` of `ZScore` transform are loaded from the `OnePassMeanVarStd` checkpoint:

> **_NOTE:_**  `cellarium-ml` does not perform any validation on the transforms being applied to the data. Please, always verify it yourself that the transforms are configured correctly. If not configured correctly, your model will silently produce wrong results. In the example below, we first apply `NormalizeTotal` and `Log1p` transforms to the data and then apply `ZScore` transform. Importantly, `mean_g` and `std_g` parameters of the `ZScore` transform were calculated using `OnePassMeanVarStd` model on the data that was also transformed with `NormalizeTotal` and `Log1p`.

```diff
<   transforms: null
---
>   transforms:
>     - class_path: cellarium.ml.transforms.NormalizeTotal
>       init_args:
>         target_count: 10_000
>     - cellarium.ml.transforms.Log1p
>     - class_path: cellarium.ml.transforms.ZScore
>       init_args:
>         mean_g:
>           !CheckpointLoader
>           file_path: /tmp/test_examples/onepass/lightning_logs/version_0/checkpoints/epoch=0-step=2.ckpt
>           attr: model.mean_g
>           convert_fn: null
>         std_g:
>           !CheckpointLoader
>           file_path: /tmp/test_examples/onepass/lightning_logs/version_0/checkpoints/epoch=0-step=2.ckpt
>           attr: model.std_g
>           convert_fn: null
>         var_names_g:
>           !CheckpointLoader
>           file_path: /tmp/test_examples/onepass/lightning_logs/version_0/checkpoints/epoch=0-step=2.ckpt
>           attr: model.var_names_g
>           convert_fn: numpy.ndarray.tolist
```

Set the number of components for `IncrementalPCA`:

```diff
<       n_components: null
---
>       n_components: 50
```

### data

Configure the `DistributedAnnDataCollection`. Here we validate `obs` columns that are used by the transforms and the model (`total_mrna_umis`):

```diff
<       filenames: null
<       shard_size: null
<       max_cache_size: 1
<       obs_columns_to_validate: null
---
>       filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad
>       shard_size: 100
>       max_cache_size: 2
>       obs_columns_to_validate:
>         - total_mrna_umis
```

Configure the `DataLoader`. `batch_keys` contains the same keys as for `OnePassMeanVarStd` above:

```diff
<   batch_keys: null
<   batch_size: 1
<   num_workers: 0
---
>   batch_keys:
>     x_ng:
>       attr: X
>       convert_fn: cellarium.ml.utilities.data.densify
>     var_names_g:
>       attr: var_names
>     total_mrna_umis_n:
>       attr: obs
>       key: total_mrna_umis
>   batch_size: 100
>   num_workers: 2
```

## `LogisticRegression`

Generate a default config file:

```
cellarium-ml logistic_regression fit --print_config > lr_train_config.yaml
```

Below we highlight the changes made to the default configuration file.

### train

Change the number of devices, change strategy to `ddp_find_unused_parameters_true` (because trained PCA model contains a dummy parameter that is not used in producing the loss during training; read more about `find_unused_parameters` option in the [DDP docs](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)), set the number of epochs, and set the path for logs and weights:

```diff
<   strategy: auto
<   devices: auto
<   max_epochs: null
<   default_root_dir: null
---
>   strategy: ddp_find_unused_parameters_true
>   devices: 2
>   max_epochs: 5
>   default_root_dir: /tmp/test_examples/lr
```

### model

Add trained PCA model as a transform. Note, that the trained PCA model contains `NormalizeTotal` and `Log1p`, and `ZScore` transforms in its pipeline:

```diff
<   transforms: null
---
>   transforms:
>     - !CheckpointLoader
>       file_path: /tmp/test_examples/ipca/lightning_logs/version_0/checkpoints/epoch=0-step=2.ckpt
>       attr: null
>       convert_fn: null
```

Set the optimizer and its learning rate:

```diff
<   optim_fn: null
<   optim_kwargs: null
---
>   optim_fn: torch.optim.Adam
>   optim_kwargs:
>     lr: 0.1
```

### data

Configure the `DistributedAnnDataCollection`. Here we validate `obs` columns that are used by the transforms and the model (`total_mrna_umis` as above and additionally `assay` column):

```diff
<       filenames: null
<       shard_size: null
<       max_cache_size: 1
<       obs_columns_to_validate: null
---
>       filenames: https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad
>       shard_size: 100
>       max_cache_size: 2
>       obs_columns_to_validate:
>         - total_mrna_umis
>         - assay
```

Configure the `DataLoader`. `batch_keys` contains the same keys as above and additionally `y_n` which is an argument to `LogisticRegression`:

```diff
<   batch_keys: null
<   batch_size: 1
<   num_workers: 0
---
>   batch_keys:
>     x_ng:
>       attr: X
>       convert_fn: cellarium.ml.utilities.data.densify
>     var_names_g:
>       attr: var_names
>     total_mrna_umis_n:
>       attr: obs
>       key: total_mrna_umis
>     y_n:
>       attr: obs
>       key: assay
>       convert_fn: cellarium.ml.utilities.data.categories_to_codes
>   batch_size: 100
>   num_workers: 2
```