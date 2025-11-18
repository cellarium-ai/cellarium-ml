# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import pickle
from pathlib import Path
from unittest.mock import patch

import lightning.pytorch as pl
import numpy as np
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models.socam import SOCAM
from cellarium.ml.utilities.data import collate_fn
from tests.common import BoringDataset


def create_mock_pickle_files(tmp_path: Path, n_categories: int = 10):
    """
    Create mock pickle files required by SOCAM model.

    Args:
        tmp_path: Temporary directory path
        n_categories: Number of categories/cell types

    Returns:
        Dictionary with paths to created pickle files
    """
    # Create y_categories (cell type labels)
    y_categories = np.array([f"cell_type_{i}" for i in range(n_categories)])
    y_categories_path = tmp_path / "y_categories.pkl"
    with open(y_categories_path, "wb") as f:
        pickle.dump(y_categories, f)

    # Create target_row_descendent_col_torch_tensor (descendant matrix for probability propagation)
    # Identity matrix for simple case - each cell type is only a descendant of itself
    descendant_matrix = torch.eye(n_categories, dtype=torch.float32)
    descendant_matrix_path = tmp_path / "descendant_matrix.pkl"
    with open(descendant_matrix_path, "wb") as f:
        pickle.dump(descendant_matrix, f)

    # Create valid_mask (boolean mask for valid categories)
    valid_mask = np.ones(n_categories, dtype=bool)
    valid_mask_path = tmp_path / "valid_mask.pkl"
    with open(valid_mask_path, "wb") as f:
        pickle.dump(valid_mask, f)

    return {
        "y_categories_path": str(y_categories_path),
        "target_row_descendent_col_torch_tensor_path": str(descendant_matrix_path),
        "valid_mask_path": str(valid_mask_path),
    }


def mock_read_pkl_from_gcs(filename: str):
    """
    Mock function to read pickle files from local filesystem instead of GCS.
    Used to replace cellarium.ml.data.fileio.read_pkl_from_gcs during testing.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    """
    Test SOCAM model checkpoint saving and loading with multi-device support.
    Similar to test_logistic_regression.py but adapted for SOCAM model.
    """
    n, g, c = 8, 5, 10  # samples, genes, categories
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    y = np.random.randint(0, c, size=n)
    devices = int(os.environ.get("TEST_DEVICES", "1"))

    # Create mock pickle files
    pickle_paths = create_mock_pickle_files(tmp_path, n_categories=c)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        BoringDataset(
            np.random.randn(n, g).astype(np.float32),
            var_names=var_names_g,
            y=y,
        ),
        collate_fn=collate_fn,
        batch_size=4,
    )

    # Patch read_pkl_from_gcs to use local files - patch where SOCAM imports it
    with patch("cellarium.ml.models.socam.read_pkl_from_gcs", side_effect=mock_read_pkl_from_gcs):
        # Model
        model = SOCAM(
            n_obs=n,
            var_names_g=var_names_g,
            actual_categories=c,
            W_prior_scale=1.0,
            W_init_scale=1.0,
            seed=42,
            probability_propagation_flag=False,
            target_row_descendent_col_torch_tensor_path=pickle_paths["target_row_descendent_col_torch_tensor_path"],
            y_categories_path=pickle_paths["y_categories_path"],
            valid_mask_path=pickle_paths["valid_mask_path"],
            log_metrics=False,
        )
        module = CellariumModule(model=model, optim_fn=torch.optim.Adam, optim_kwargs={"lr": 1e-3})

        # Trainer
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=devices,
            max_epochs=1,
            default_root_dir=tmp_path,
        )

        # Fit
        trainer.fit(module, train_dataloaders=train_loader)

        # Run tests only for rank 0
        if trainer.global_rank != 0:
            return

        # Load model from checkpoint
        ckpt_path = tmp_path / f"lightning_logs/version_0/checkpoints/epoch=0-step={math.ceil(n / devices / 4)}.ckpt"
        assert ckpt_path.is_file()

        # Load the checkpoint with the same mock
        loaded_module = CellariumModule.load_from_checkpoint(ckpt_path)
        loaded_model = loaded_module.model
        assert isinstance(loaded_model, SOCAM)

        # Assert model attributes match
        assert np.array_equal(model.var_names_g, loaded_model.var_names_g)
        assert np.array_equal(model.y_categories, loaded_model.y_categories)
        assert model.n_categories == loaded_model.n_categories
        assert model.actual_categories == loaded_model.actual_categories
        assert model.probability_propagation_flag == loaded_model.probability_propagation_flag

        # Test prediction from loaded checkpoint
        test_x_ng = torch.randn(2, g)
        output: dict[str, np.ndarray | torch.Tensor] = loaded_model.predict(test_x_ng, var_names_g)

        # Assert prediction output structure
        assert "y_logits_nc" in output
        assert "cell_type_probs_nc" in output
        assert isinstance(output["y_logits_nc"], torch.Tensor)
        assert isinstance(output["cell_type_probs_nc"], torch.Tensor)
        assert output["y_logits_nc"].shape == (2, c)
        assert output["cell_type_probs_nc"].shape == (2, c)

        # Assert probabilities are valid
        assert torch.all(output["cell_type_probs_nc"] >= 0)
        assert torch.all(output["cell_type_probs_nc"] <= 1)
        prob_sums = output["cell_type_probs_nc"].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(2), atol=1e-5)


def test_socam_predict(tmp_path: Path):
    """
    Test SOCAM predict method to ensure it returns correct output structure.
    """
    n, g, c = 4, 3, 5
    var_names_g = np.array([f"gene_{i}" for i in range(g)])

    # Create mock pickle files
    pickle_paths = create_mock_pickle_files(tmp_path, n_categories=c)

    # Patch read_pkl_from_gcs to use local files - patch where SOCAM imports it
    with patch("cellarium.ml.models.socam.read_pkl_from_gcs", side_effect=mock_read_pkl_from_gcs):
        # Model
        model = SOCAM(
            n_obs=n,
            var_names_g=var_names_g,
            actual_categories=c,
            probability_propagation_flag=False,
            target_row_descendent_col_torch_tensor_path=pickle_paths["target_row_descendent_col_torch_tensor_path"],
            y_categories_path=pickle_paths["y_categories_path"],
            valid_mask_path=pickle_paths["valid_mask_path"],
            log_metrics=False,
        )

        # Create test data
        x_ng = torch.randn(n, g)

        # Run prediction
        output = model.predict(x_ng, var_names_g)

        # Assert output structure
        assert "y_logits_nc" in output
        assert "cell_type_probs_nc" in output

        # Assert output shapes
        assert output["y_logits_nc"].shape == (n, c)
        assert output["cell_type_probs_nc"].shape == (n, c)

        # Assert probabilities sum to ~1 for each sample
        assert isinstance(output["cell_type_probs_nc"], torch.Tensor)
        prob_sums = output["cell_type_probs_nc"].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(n), atol=1e-5)

        # Assert all probabilities are in [0, 1]
        assert torch.all(output["cell_type_probs_nc"] >= 0)
        assert torch.all(output["cell_type_probs_nc"] <= 1)


def test_socam_probability_propagation(tmp_path: Path):
    """
    Test SOCAM with probability propagation enabled.
    """
    n, g, c = 4, 3, 5
    var_names_g = np.array([f"gene_{i}" for i in range(g)])

    # Create mock pickle files with a hierarchical descendant matrix
    # For example: cell_type_0 is ancestor of cell_type_1
    y_categories = np.array([f"cell_type_{i}" for i in range(c)])
    y_categories_path = tmp_path / "y_categories.pkl"
    with open(y_categories_path, "wb") as f:
        pickle.dump(y_categories, f)

    # Create a simple hierarchy: type 0 -> type 1, type 2 -> type 3, type 4 standalone
    descendant_matrix = torch.eye(c, dtype=torch.float32)
    descendant_matrix[0, 1] = 1.0  # type 0 includes type 1 as descendant
    descendant_matrix[2, 3] = 1.0  # type 2 includes type 3 as descendant
    descendant_matrix_path = tmp_path / "descendant_matrix.pkl"
    with open(descendant_matrix_path, "wb") as f:
        pickle.dump(descendant_matrix, f)

    valid_mask = np.ones(c, dtype=bool)
    valid_mask_path = tmp_path / "valid_mask.pkl"
    with open(valid_mask_path, "wb") as f:
        pickle.dump(valid_mask, f)

    pickle_paths = {
        "y_categories_path": str(y_categories_path),
        "target_row_descendent_col_torch_tensor_path": str(descendant_matrix_path),
        "valid_mask_path": str(valid_mask_path),
    }

    # Patch read_pkl_from_gcs to use local files - patch where SOCAM imports it
    with patch("cellarium.ml.models.socam.read_pkl_from_gcs", side_effect=mock_read_pkl_from_gcs):
        # Model with probability propagation
        model = SOCAM(
            n_obs=n,
            var_names_g=var_names_g,
            actual_categories=c,
            probability_propagation_flag=True,
            target_row_descendent_col_torch_tensor_path=pickle_paths["target_row_descendent_col_torch_tensor_path"],
            y_categories_path=pickle_paths["y_categories_path"],
            valid_mask_path=pickle_paths["valid_mask_path"],
            log_metrics=False,
        )

        # Create test data
        x_ng = torch.randn(n, g)

        # Run prediction with probability propagation
        output = model.predict(x_ng, var_names_g)

        # Assert output structure
        assert "y_logits_nc" in output
        assert "cell_type_probs_nc" in output

        # Assert output shapes
        assert output["y_logits_nc"].shape == (n, c)
        assert output["cell_type_probs_nc"].shape == (n, c)

        # Assert all probabilities are in [0, 1] and clamped at max 1.0
        assert isinstance(output["cell_type_probs_nc"], torch.Tensor)
        assert torch.all(output["cell_type_probs_nc"] >= 0)
        assert torch.all(output["cell_type_probs_nc"] <= 1)

        # Test probability propagation logic:
        # Calculate what the probabilities would be WITHOUT propagation
        logits_nc = x_ng @ model.W_gc + model.b_c
        probs_no_propagation = torch.nn.functional.softmax(logits_nc, dim=1)

        # Manually compute expected propagated probabilities
        # According to the descendant matrix:
        # - Type 0 should get: prob(type 0) + prob(type 1)
        # - Type 1 should get: prob(type 1) [only itself]
        # - Type 2 should get: prob(type 2) + prob(type 3)
        # - Type 3 should get: prob(type 3) [only itself]
        # - Type 4 should get: prob(type 4) [only itself]
        expected_propagated = torch.zeros_like(probs_no_propagation)
        expected_propagated[:, 0] = probs_no_propagation[:, 0] + probs_no_propagation[:, 1]  # Type 0 includes type 1
        expected_propagated[:, 1] = probs_no_propagation[:, 1]  # Type 1 only itself
        expected_propagated[:, 2] = probs_no_propagation[:, 2] + probs_no_propagation[:, 3]  # Type 2 includes type 3
        expected_propagated[:, 3] = probs_no_propagation[:, 3]  # Type 3 only itself
        expected_propagated[:, 4] = probs_no_propagation[:, 4]  # Type 4 only itself

        # Clamp expected values at 1.0 (as done in the actual implementation)
        expected_propagated = torch.clamp(expected_propagated, max=1.0)

        # Assert propagated probabilities match expected values
        assert torch.allclose(output["cell_type_probs_nc"], expected_propagated, atol=1e-5), (
            f"Probability propagation failed.\nExpected:\n{expected_propagated}\nGot:\n{output['cell_type_probs_nc']}"
        )

        # Verify that parent types have higher or equal probability than before propagation
        # (since they now include their descendants)
        assert torch.all(output["cell_type_probs_nc"][:, 0] >= probs_no_propagation[:, 0]), (
            "Type 0 (parent) should have prob >= original prob"
        )
        assert torch.all(output["cell_type_probs_nc"][:, 2] >= probs_no_propagation[:, 2]), (
            "Type 2 (parent) should have prob >= original prob"
        )
