Example usage of `scvi-distributed` infrastructure for distributed training

## ddp_distributed_sampler.py
Using the DistributedAnnDataCollection, uses the DistributedAnnDataCollectionSampler with pytorch lightning autoencoder to show distributed (multi-GPU) and training with num_workers > 1

## ddp_iterable_dataset.py
Without using the DistributedAnnDataCollection, an iterable-dataset approach with support for multi-process (GPU) and multi-worker sampling inside of the Dataset itself.  