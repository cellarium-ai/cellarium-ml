# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.models import ImputationModel
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes, collate_fn, densify
from tests.common import BoringDatasetSCVI


@pytest.mark.parametrize("n_batch", [2, 4], ids=lambda s: f"n_batch_{s}")
@pytest.mark.parametrize("masking_probability", [0.3, 0.5], ids=["mask_30", "mask_50"])
def test_load_from_checkpoint_multi_device(
    n_batch: int,
    masking_probability: float,
    tmp_path: Path,
):
    """Test loading ImputationModel from checkpoint on multiple devices."""
    n, g = 32, 20
    batch_size = 8  # must be > 1 for BatchNorm
    var_names_g = [f"gene_{i}" for i in range(g)]
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDatasetSCVI(
            data=np.random.poisson(lam=2.0, size=(n, g)),
            batch_index_n=np.random.randint(0, n_batch, size=n),
            var_names=np.array(var_names_g),
        ),
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    
    # model
    model = ImputationModel(
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=10,
        masking_probability=masking_probability,
        encoder={
            "hidden_layers": [],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
        },
        decoder={
            "hidden_layers": [],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
            "final_additive_bias": False,
        },
    )
    
    module = CellariumModule(model=model)
    
    # trainer
    import lightning.pytorch as pl
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=devices,
        max_epochs=1,
        default_root_dir=tmp_path,
        strategy=DDPStrategy(find_unused_parameters=True) if devices > 1 else "auto",
    )
    
    # fit
    trainer.fit(module, train_loader)
    
    # save checkpoint
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    
    # load from checkpoint
    loaded_module = CellariumModule.load_from_checkpoint(checkpoint_path)
    
    # test that the loaded model has the same configuration
    assert isinstance(loaded_module.model, ImputationModel)
    assert loaded_module.model.masking_probability == masking_probability
    assert loaded_module.model.n_batch == n_batch
    assert len(loaded_module.model.var_names_g) == g


def test_run():
    """Test basic functionality of ImputationModel with fake data."""
    n, g = 32, 20
    n_batch = 2
    batch_size = 8
    var_names_g = [f"gene_{i}" for i in range(g)]
    masking_probability = 0.4
    
    # create fake data
    data = np.random.poisson(lam=2.0, size=(n, g))
    batch_indices = np.random.randint(0, n_batch, size=n)
    
    # create dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDatasetSCVI(
            data=data,
            batch_index_n=batch_indices,
            var_names=np.array(var_names_g),
        ),
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    
    # create model
    model = ImputationModel(
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=8,
        masking_probability=masking_probability,
        encoder={
            "hidden_layers": [
                {
                    "class_path": "torch.nn.Linear",
                    "init_args": {"out_features": 16},
                }
            ],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
        },
        decoder={
            "hidden_layers": [
                {
                    "class_path": "torch.nn.Linear", 
                    "init_args": {"out_features": 16},
                }
            ],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
            "final_additive_bias": False,
        },
    )
    
    # test forward pass
    model.train()
    for batch in train_loader:
        output = model(
            x_ng=batch["x_ng"].float(),
            var_names_g=batch["var_names_g"],
            batch_index_n=batch["batch_index_n"],
        )
        
        # check that all required keys are in output
        required_keys = ["loss", "reconstruction_loss", "kl_divergence_z", "z_nk"]
        for key in required_keys:
            assert key in output, f"Missing key {key} in model output"
        
        # check tensor shapes
        batch_size_actual = batch["x_ng"].shape[0]
        assert output["loss"].shape == torch.Size([]), "Loss should be scalar"
        assert output["reconstruction_loss"].shape == torch.Size([batch_size_actual]), \
            f"Reconstruction loss should have batch dimension {batch_size_actual}"
        assert output["kl_divergence_z"].shape == torch.Size([batch_size_actual]), \
            f"KL divergence should have batch dimension {batch_size_actual}"
        assert output["z_nk"].shape == torch.Size([batch_size_actual, 8]), \
            f"Latent representation should be [{batch_size_actual}, 8]"
        
        # check that loss is finite
        assert torch.isfinite(output["loss"]), "Loss should be finite"
        assert torch.all(torch.isfinite(output["reconstruction_loss"])), \
            "All reconstruction losses should be finite"
        assert torch.all(torch.isfinite(output["kl_divergence_z"])), \
            "All KL divergences should be finite"
        
        # test that masking is working - we can't directly test the mask
        # but we can ensure the forward pass completes without error
        break  # only test one batch
    
    # test evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            output = model(
                x_ng=batch["x_ng"].float(),
                var_names_g=batch["var_names_g"],
                batch_index_n=batch["batch_index_n"],
            )
            
            # same checks as above
            required_keys = ["loss", "reconstruction_loss", "kl_divergence_z", "z_nk"]
            for key in required_keys:
                assert key in output, f"Missing key {key} in model output"
            
            assert torch.isfinite(output["loss"]), "Loss should be finite in eval mode"
            break
    
    print("ImputationModel test passed successfully!")


@pytest.mark.parametrize("masking_probability", [0.2, 0.5, 0.8], ids=["mask_20", "mask_50", "mask_80"])
def test_masking_functionality(masking_probability: float):
    """Test that the masking functionality works as expected."""
    n, g = 16, 10
    n_batch = 2
    var_names_g = [f"gene_{i}" for i in range(g)]
    
    # create fake data with non-zero values so we can detect masking
    data = np.ones((n, g)) * 5.0  # all genes have value 5
    batch_indices = np.random.randint(0, n_batch, size=n)
    
    # create model
    model = ImputationModel(
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=4,
        masking_probability=masking_probability,
        encoder={
            "hidden_layers": [],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
        },
        decoder={
            "hidden_layers": [],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
            "final_additive_bias": False,
        },
    )
    
    # set fixed random seed for reproducible masking
    torch.manual_seed(42)
    
    # test data
    x_ng = torch.tensor(data, dtype=torch.float32)
    
    # test the mask creation method directly
    mask = model.create_gene_mask(g, x_ng.device)
    
    # verify the correct number of genes were masked
    expected_masked_genes = int(masking_probability * g)
    actual_masked_genes = mask.sum().item()
    assert actual_masked_genes == expected_masked_genes, \
        f"Expected {expected_masked_genes} genes to be masked, but {actual_masked_genes} were masked"
    
    # test the mask application method
    masked_data = model.apply_gene_mask(x_ng, mask)
    
    # verify that masked genes are set to zero
    masked_positions = mask.nonzero(as_tuple=True)[0]
    unmasked_positions = (~mask).nonzero(as_tuple=True)[0]
    
    # check that masked positions are zero in all samples
    assert torch.all(masked_data[:, masked_positions] == 0.0), \
        "Masked gene positions should be set to zero"
    
    # check that unmasked positions retain original values
    assert torch.allclose(masked_data[:, unmasked_positions], x_ng[:, unmasked_positions]), \
        "Unmasked gene positions should retain original values"
    
    # verify that the mask is consistent across all samples in the batch
    # (same genes are masked for all cells in a batch)
    for i in range(1, masked_data.shape[0]):
        assert torch.equal(masked_data[i, masked_positions], torch.zeros_like(masked_data[i, masked_positions])), \
            f"Masked positions should be zero for all samples in batch, failed at sample {i}"
    
    # test the full forward pass to ensure integration works
    var_names_g_array = np.array(var_names_g)
    batch_index_n = torch.tensor(batch_indices, dtype=torch.long)
    
    model.train()
    output = model(
        x_ng=x_ng,
        var_names_g=var_names_g_array,
        batch_index_n=batch_index_n,
    )
    
    # verify forward pass completes successfully
    required_keys = ["loss", "reconstruction_loss", "kl_divergence_z", "z_nk"]
    for key in required_keys:
        assert key in output, f"Missing key {key} in model output"
    
    print(f"Masking test passed! Masked {actual_masked_genes}/{g} genes with probability {masking_probability}")


def test_different_seeds_produce_different_masks():
    """Test that different random seeds produce different masks."""
    n, g = 8, 20
    n_batch = 2
    var_names_g = [f"gene_{i}" for i in range(g)]
    masking_probability = 0.5
    
    # create fake data
    data = np.ones((n, g)) * 3.0
    batch_indices = np.random.randint(0, n_batch, size=n)
    
    # create model
    model = ImputationModel(
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=4,
        masking_probability=masking_probability,
        encoder={
            "hidden_layers": [],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
        },
        decoder={
            "hidden_layers": [],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
            "final_additive_bias": False,
        },
    )
    
    x_ng = torch.tensor(data, dtype=torch.float32)
    
    # capture masks with different seeds using the refactored method
    masks = []
    for seed in [123, 456, 789]:
        torch.manual_seed(seed)
        mask = model.create_gene_mask(g, x_ng.device)
        masks.append(mask.clone())
    
    # verify that different seeds produce different masks
    assert not torch.equal(masks[0], masks[1]), "Different seeds should produce different masks"
    assert not torch.equal(masks[1], masks[2]), "Different seeds should produce different masks"
    assert not torch.equal(masks[0], masks[2]), "Different seeds should produce different masks"
    
    # verify all masks have the same number of masked genes
    expected_masked = int(masking_probability * g)
    for i, mask in enumerate(masks):
        assert mask.sum().item() == expected_masked, \
            f"Mask {i} should have {expected_masked} masked genes, got {mask.sum().item()}"
    
    print("Different seeds test passed! Each seed produces a unique mask pattern.")


def test_mask_creation_and_application_methods():
    """Test the individual create_gene_mask and apply_gene_mask methods."""
    g = 12
    n = 8
    masking_probability = 0.5
    var_names_g = [f"gene_{i}" for i in range(g)]
    
    # create model
    model = ImputationModel(
        var_names_g=var_names_g,
        n_batch=2,
        n_latent=4,
        masking_probability=masking_probability,
        encoder={
            "hidden_layers": [],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
        },
        decoder={
            "hidden_layers": [],
            "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
            "final_additive_bias": False,
        },
    )
    
    # test create_gene_mask method
    torch.manual_seed(100)
    device = torch.device("cpu")
    mask1 = model.create_gene_mask(g, device)
    
    # verify mask properties
    assert mask1.dtype == torch.bool, "Mask should be boolean tensor"
    assert mask1.shape == (g,), f"Mask should have shape ({g},), got {mask1.shape}"
    expected_masked = int(masking_probability * g)
    assert mask1.sum().item() == expected_masked, \
        f"Should mask {expected_masked} genes, got {mask1.sum().item()}"
    
    # test repeatability with same seed
    torch.manual_seed(100)
    mask2 = model.create_gene_mask(g, device)
    assert torch.equal(mask1, mask2), "Same seed should produce identical masks"
    
    # test apply_gene_mask method
    x_ng = torch.ones((n, g)) * 7.0  # all values are 7
    masked_x = model.apply_gene_mask(x_ng, mask1)
    
    # verify masking application
    assert masked_x.shape == x_ng.shape, "Masked data should have same shape as input"
    masked_positions = mask1.nonzero(as_tuple=True)[0]
    unmasked_positions = (~mask1).nonzero(as_tuple=True)[0]
    
    # check masked positions are zero
    assert torch.all(masked_x[:, masked_positions] == 0.0), \
        "Masked positions should be zero"
    
    # check unmasked positions are preserved
    assert torch.allclose(masked_x[:, unmasked_positions], x_ng[:, unmasked_positions]), \
        "Unmasked positions should preserve original values"
    
    # verify original data is unchanged
    assert torch.all(x_ng == 7.0), "Original data should be unchanged"
    
    print("Individual masking methods test passed!")
