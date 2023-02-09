Example usage of `scvi-distributed` infrastructure for distributed training

## ddp_distributed_sampler.py
Using the DistributedAnnDataCollection, uses the DistributedAnnDataCollectionSampler with pytorch lightning autoencoder to show distributed (multi-GPU) and training with num_workers > 1

To execute run: `python examples/ddp_distributed_sampler.py`