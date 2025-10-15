2024.11.08

- preliminary small run
- we downloaded 100 files from gs://cellarium-file-system/curriculum/human_cellariumgpt_v2/extract_files
    - to a local directory
- we computed highly variable genes
    - cellarium-ml onepass_mean_var_std fit -c onepass_train_config.yaml
    - then run the notebook compute_hvg.ipynb which uses the function cellarium.ml.preprocessing.get_highly_variable_genes
- we ran scvi fit on that local data subset
    - cellarium-ml scvi fit -c scvi_train_config.yaml
