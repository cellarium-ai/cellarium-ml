# Big run of scVI in cellarium

Some notes. Lessons learned and open questions.

## Question: scvi-tools CZI run config
- we were thinking of doing an initial run to match their settings
- https://github.com/chanzuckerberg/cellxgene-census/blob/main/tools/models/scvi/scvi-config.yaml
- we are confused because when you download their model by doing

    ```console
    aws s3 cp --no-sign-request --no-progress --only-show-errors s3://cellxgene-contrib-public/models/scvi/2025-01-30/homo_sapiens/model.pt 2025-01-30-scvi-homo-sapiens/scvi.model/
    ```
    then the model (loaded this way)
    ```python
    import scvi
    import anndata
    # some h5ad from cellxgene census
    adata = anndata.read_h5ad('UBERON_0002115.h5ad')
    # make sure there is a "batch" key, doesn't matter what for this
    adata.obs['batch'] = adata.obs['combo']
    # load the model
    dir_with_model_pt = '2025-01-30-scvi-homo-sapiens/scvi.model'
    scvi.model.SCVI.prepare_query_anndata(adata, dir_with_model_pt)
    vae_q = scvi.model.SCVI.load_query_data(adata, dir_with_model_pt)
    # print the model architecture
    vae_q.module
    ```
    seems to use a hidden dimension of only 128, not 512.
    In fact, here it is:

    ```
    VAE(
      (z_encoder): Encoder(
        (encoder): FCLayers(
          (fc_layers): Sequential(
            (Layer 0): Sequential(
              (0): Linear(in_features=8000, out_features=128    bias=True)
              (1): BatchNorm1d(128, eps=0.001, momentum=0    affine=True,         track_running_stats=True)
              (2): None
              (3): ReLU()
              (4): Dropout(p=0.1, inplace=False)
            )
            (Layer 1): Sequential(
              (0): Linear(in_features=128, out_features=128    bias=True)
              (1): BatchNorm1d(128, eps=0.001, momentum=0    affine=True,         track_running_stats=True)
              (2): None
              (3): ReLU()
              (4): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (mean_encoder): Linear(in_features=128, out_features=50    bias=True)
        (var_encoder): Linear(in_features=128, out_features=50    bias=True)
      )
      (l_encoder): Encoder(
        (encoder): FCLayers(
          (fc_layers): Sequential(
            (Layer 0): Sequential(
              (0): Linear(in_features=8000, out_features=128    bias=True)
              (1): BatchNorm1d(128, eps=0.001, momentum=0.01    affine=True,         track_running_stats=True)
              (2): None
              (3): ReLU()
              (4): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (mean_encoder): Linear(in_features=128, out_features=1    bias=True)
        (var_encoder): Linear(in_features=128, out_features=1    bias=True)
      )
      (decoder): DecoderSCVI(
        (px_decoder): FCLayers(
          (fc_layers): Sequential(
            (Layer 0): Sequential(
              (0): Linear(in_features=11158, out_features=128    bias=True)
              (1): BatchNorm1d(128, eps=0.001, momentum=0.01    affine=True,         track_running_stats=True)
              (2): None
              (3): ReLU()
              (4): None
            )
            (Layer 1): Sequential(
              (0): Linear(in_features=11236, out_features=128    bias=True)
              (1): BatchNorm1d(128, eps=0.001, momentum=0.01    affine=True,         track_running_stats=True)
              (2): None
              (3): ReLU()
              (4): None
            )
          )
        )
        (px_scale_decoder): Sequential(
          (0): Linear(in_features=128, out_features=8000, bias=True)
          (1): Softmax(dim=-1)
        )
        (px_r_decoder): Linear(in_features=128, out_features=8000    bias=True)
        (px_dropout_decoder): Linear(in_features=128    out_features=8000,         bias=True)
      )
    )
    ```
    they are also using an `l_encoder` which is not what I thought. `use_observed_lib_size = True` is the default, right?

## Learned: KL annealing schedule
- critical for improving performance
- trying their settings

    https://github.com/chanzuckerberg/cellxgene-census/blob/19a2ca31ad566f1a128f4555d23e468feff1ae4f/tools/models/scvi/scvi-config.yaml#L34-L36

## Learned: speed optimizations
- the model is tiny
- thoughput is all about dataloading
- have to have data on a local SSD (or SSD boot disk) to get any kind of throughput. just download the whole curriculum.
    - should pre-filter the h5ad files to only include genes being used
        - nice helper script for this exists. can filter 4440 files in 1.5 hours.
        - going from 33k genes to 3k genes speeds up anndata.read_h5ad by a factor of 100 (sensible)
            - same for backed mode, I checked. I guess adata[:, slice].to_memory() fetches all in memory and then slices.
                - probably a limitation of CSR format?
- to keep a single T4 even the slightest bit occupied, you need like 14 workers throwing data at it as fast as possible
    - use prefetch
    - cache doesn't matter much but should probably equal prefetch for prefetch to actually be doing anything?
- reasonable settings for an n1-standard-16 with T4:
    - num_workers: 14
    - max_cache_size: 3 (matches prefetch)
    - cache_size_strictly_enforced: true (this helps prevent memory spikes)
    - prefetch_factor: 3
    - persistent_workers: true
    - the above is for batch_size: 1024
    - this can crank through a single epoch of 44M cells in 40 mins
        - files have 3241 genes

## Question: the model
- they use a model that has "batch" being the concatenation of

    - dataset_id
    - assay
    - suspension_type
    - donor_id

    https://github.com/chanzuckerberg/cellxgene-census/blob/19a2ca31ad566f1a128f4555d23e468feff1ae4f/tools/models/scvi/scvi-config.yaml#L16-L17

- we were just going to do [dataset_id, donor_id]
    - unclear if datasets exist where one (dataset, donor) is profiled
      with more than one suspension/assay. Had assumed "no" but realized only now that this is an assumption.
    - was thinking about using categorical covariates for:
        - assay
        - suspension_type
    - categorical covariates make a great deal of sense to me for handling assay and suspension type

- the scvi-tools CZI run uses no batch labels in the encoder at all
    - it injects batch labels at every layer of the decoder

## Learned: the data
- they impose a filter which only includes cells with >= 300 genes
- this is quite a stringent cutoff which excludes a lot of cells in cellxgene

    https://github.com/chanzuckerberg/cellxgene-census/blob/19a2ca31ad566f1a128f4555d23e468feff1ae4f/tools/models/scvi/scvi-config.yaml#L7

## Question: convergence
- it's unclear whether training for 100 epochs is at all necessary
- training loss pretty much flat-lines after only a few epochs
