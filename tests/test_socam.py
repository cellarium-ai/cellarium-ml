# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.models.socam import SOCAM, compute_valid_mask
from cellarium.ml.utilities.data import collate_fn


def test_compute_valid_mask_identical_categories():
    """Test when input and output categories are identical."""
    categories = ["a", "b", "c"]
    result = compute_valid_mask(input_categories=categories, output_categories=categories)
    assert result == [0, 1, 2]


def test_compute_valid_mask_subset_categories():
    """Test when input categories are a subset of output categories."""
    input_categories = ["b", "d"]
    output_categories = ["a", "b", "c", "d", "e"]
    result = compute_valid_mask(input_categories=input_categories, output_categories=output_categories)
    assert result == [1, 3]


def test_compute_valid_mask_reordered_categories():
    """Test when input categories are reordered relative to output."""
    input_categories = ["c", "a", "b"]
    output_categories = ["a", "b", "c"]
    result = compute_valid_mask(input_categories=input_categories, output_categories=output_categories)
    assert result == [2, 0, 1]


def test_compute_valid_mask_single_category():
    """Test with a single input category."""
    input_categories = ["b"]
    output_categories = ["a", "b", "c"]
    result = compute_valid_mask(input_categories=input_categories, output_categories=output_categories)
    assert result == [1]


def test_compute_valid_mask_missing_category_raises_error():
    """Test that missing category in output raises KeyError."""
    input_categories = ["a", "x"]
    output_categories = ["a", "b", "c"]
    with pytest.raises(KeyError):
        compute_valid_mask(input_categories=input_categories, output_categories=output_categories)


def test_load_from_checkpoint_multi_device(tmp_path: Path):
    """
    Test SOCAM model checkpoint saving and loading with multi-device support.
    """
    n, g, c = 8, 5, 10  # samples, genes, categories
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    y = np.array([f"cell_type_{i}" for i in np.random.randint(0, c, size=n)])
    devices = int(os.environ.get("TEST_DEVICES", "1"))

    # Create test data
    output_categories = [f"cell_type_{i}" for i in range(c)]
    descendant_matrix = torch.eye(c, dtype=torch.float32)

    # Inline dataset that emits cl_names_n (string labels)
    class SOCAMDataset(torch.utils.data.Dataset):
        def __init__(self, x: np.ndarray, var_names: np.ndarray, cl_names: np.ndarray) -> None:
            self.x = x
            self.var_names = var_names
            self.cl_names = cl_names

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
            return {"x_ng": self.x[idx, None], "var_names_g": self.var_names, "cl_names_n": self.cl_names[idx, None]}

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        SOCAMDataset(
            np.random.randn(n, g).astype(np.float32),
            var_names=var_names_g,
            cl_names=y,
        ),
        collate_fn=collate_fn,
        batch_size=4,
    )

    # Model
    model = SOCAM(
        n_obs=n,
        var_names_g=var_names_g,
        descendant_tensor=descendant_matrix,
        cl_names=output_categories,
        W_prior_scale=1.0,
        W_init_scale=1.0,
        seed=42,
        probability_propagation_flag=False,
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

    # Load the checkpoint
    loaded_module = CellariumModule.load_from_checkpoint(ckpt_path)
    loaded_model = loaded_module.model
    assert isinstance(loaded_model, SOCAM)

    # Assert model attributes match
    assert np.array_equal(model.var_names_g, loaded_model.var_names_g)
    assert model.cl_names == loaded_model.cl_names
    assert model.n_categories == loaded_model.n_categories
    # assert model.n_output_categories == loaded_model.n_output_categories
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


def test_socam_predict():
    """
    Test SOCAM predict method to ensure it returns correct output structure.
    """
    n, g, c = 4, 3, 5
    var_names_g = np.array([f"gene_{i}" for i in range(g)])

    # Create test data
    output_categories = [f"cell_type_{i}" for i in range(c)]
    descendant_matrix = torch.eye(c, dtype=torch.float32)

    # Model
    model = SOCAM(
        n_obs=n,
        var_names_g=var_names_g,
        descendant_tensor=descendant_matrix,
        cl_names=output_categories,
        probability_propagation_flag=False,
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


def test_socam_probability_propagation():
    """
    Test SOCAM with probability propagation enabled.
    """
    n, g, c = 4, 3, 5
    var_names_g = np.array([f"gene_{i}" for i in range(g)])

    # Create test data with a hierarchical descendant matrix
    output_categories = [f"cell_type_{i}" for i in range(c)]

    # Create a simple hierarchy: type 0 -> type 1, type 2 -> type 3, type 4 standalone
    descendant_matrix = torch.eye(c, dtype=torch.float32)
    descendant_matrix[0, 1] = 1.0  # type 0 includes type 1 as descendant
    descendant_matrix[2, 3] = 1.0  # type 2 includes type 3 as descendant

    # Model with probability propagation
    model = SOCAM(
        n_obs=n,
        var_names_g=var_names_g,
        descendant_tensor=descendant_matrix,
        cl_names=output_categories,
        probability_propagation_flag=True,
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


# ---------------------------------------------------------------------------
# Helpers for subset tests
# ---------------------------------------------------------------------------


def _make_socam(
    n: int = 4,
    g: int = 3,
    c: int = 5,
    probability_propagation_flag: bool = False,
    cl_name_subset: list[str] | None = None,
) -> SOCAM:
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    cl_names = [f"cell_type_{i}" for i in range(c)]
    descendant_tensor = torch.eye(c, dtype=torch.float32)
    return SOCAM(
        n_obs=n,
        var_names_g=var_names_g,
        descendant_tensor=descendant_tensor,
        cl_names=cl_names,
        cl_name_subset=cl_name_subset,
        probability_propagation_flag=probability_propagation_flag,
        log_metrics=False,
    )


# ---------------------------------------------------------------------------
# _get_subset_info tests
# ---------------------------------------------------------------------------


def test_get_subset_info_basic():
    model = _make_socam()
    names, indices, desc, label_lookup = model._get_subset_info(["cell_type_2", "cell_type_4"])
    assert names == ["cell_type_2", "cell_type_4"]
    assert indices == [2, 4]
    assert desc.shape == (2, 2)
    # submatrix of identity should also be identity
    assert torch.equal(desc, torch.eye(2))
    assert label_lookup == {"cell_type_2": 0, "cell_type_4": 1}


def test_get_subset_info_order_independent():
    model = _make_socam()
    names_a, indices_a, desc_a, lookup_a = model._get_subset_info(["cell_type_4", "cell_type_2"])
    names_b, indices_b, desc_b, lookup_b = model._get_subset_info(["cell_type_2", "cell_type_4"])
    assert names_a == names_b
    assert indices_a == indices_b
    assert desc_a is desc_b  # same cached object
    assert lookup_a is lookup_b  # same cached dict object


def test_get_subset_info_caching():
    model = _make_socam()
    _, _, desc_first, lookup_first = model._get_subset_info(["cell_type_0", "cell_type_3"])
    _, _, desc_second, lookup_second = model._get_subset_info(["cell_type_0", "cell_type_3"])
    assert desc_first is desc_second
    assert lookup_first is lookup_second


def test_get_subset_info_invalid_name():
    model = _make_socam()
    with pytest.raises(KeyError):
        model._get_subset_info(["cell_type_0", "nonexistent"])


# ---------------------------------------------------------------------------
# forward() subset tests
# ---------------------------------------------------------------------------


def test_forward_with_cl_name_subset():
    n, g = 4, 3
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    # Sorted subset is ["cell_type_1", "cell_type_2", "cell_type_3"]
    model = _make_socam(n=n, g=g, c=5, cl_name_subset=["cell_type_3", "cell_type_1", "cell_type_2"])
    x_ng = torch.randn(n, g)
    cl_names_n = np.array(["cell_type_1", "cell_type_3", "cell_type_2", "cell_type_1"])
    result = model.forward(x_ng, var_names_g, cl_names_n)
    assert "loss" in result
    assert result["loss"] is not None
    assert result["loss"].shape == torch.Size([])  # scalar


def test_forward_no_cl_name_subset():
    n, g, c = 4, 3, 5
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    model = _make_socam(n=n, g=g, c=c)
    x_ng = torch.randn(n, g)
    cl_names_n = np.array([f"cell_type_{i}" for i in np.random.randint(0, c, size=n)])
    result = model.forward(x_ng, var_names_g, cl_names_n)
    assert "loss" in result
    assert result["loss"] is not None
    assert result["loss"].shape == torch.Size([])


# ---------------------------------------------------------------------------
# predict() subset tests
# ---------------------------------------------------------------------------


def test_predict_with_cl_name_subset():
    n, g = 4, 3
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    model = _make_socam(n=n, g=g, c=5, cl_name_subset=["cell_type_0", "cell_type_2", "cell_type_4"])
    x_ng = torch.randn(n, g)
    output = model.predict(x_ng, var_names_g)
    assert isinstance(output["y_logits_nc"], torch.Tensor)
    assert isinstance(output["cell_type_probs_nc"], torch.Tensor)
    assert output["y_logits_nc"].shape == (n, 3)
    assert output["cell_type_probs_nc"].shape == (n, 3)
    assert torch.all(output["cell_type_probs_nc"] >= 0)
    assert torch.all(output["cell_type_probs_nc"] <= 1)
    assert torch.allclose(output["cell_type_probs_nc"].sum(dim=1), torch.ones(n), atol=1e-5)


def test_predict_no_cl_name_subset():
    n, g, c = 4, 3, 5
    var_names_g = np.array([f"gene_{i}" for i in range(g)])
    model = _make_socam(n=n, g=g, c=c)
    x_ng = torch.randn(n, g)
    output = model.predict(x_ng, var_names_g)
    assert output["y_logits_nc"].shape == (n, c)
    assert output["cell_type_probs_nc"].shape == (n, c)


# ---------------------------------------------------------------------------
# _cl_names_to_indices tests
# ---------------------------------------------------------------------------


def test_cl_names_to_indices_basic():
    model = _make_socam()
    _, _, _, label_lookup = model._get_subset_info(["cell_type_0", "cell_type_2", "cell_type_4"])
    cl_names_n = np.array(["cell_type_4", "cell_type_0", "cell_type_2", "cell_type_0"])
    result = model._cl_names_to_indices(cl_names_n, label_lookup)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.long
    assert result.tolist() == [2, 0, 1, 0]  # sorted subset: 0=ct0, 1=ct2, 2=ct4


def test_cl_names_to_indices_full_lookup():
    model = _make_socam()
    # Trigger caching of full lookup via forward (no subset)
    var_names_g = np.array([f"gene_{i}" for i in range(3)])
    x_ng = torch.randn(4, 3)
    cl_names_n = np.array(["cell_type_0", "cell_type_1", "cell_type_2", "cell_type_3"])
    model.forward(x_ng, var_names_g, cl_names_n)
    assert model._full_label_lookup is not None
    result = model._cl_names_to_indices(cl_names_n, model._full_label_lookup)
    assert result.tolist() == [0, 1, 2, 3]


def test_cl_names_to_indices_invalid_name():
    model = _make_socam()
    _, _, _, label_lookup = model._get_subset_info(["cell_type_0", "cell_type_1"])
    cl_names_n = np.array(["cell_type_0", "cell_type_99"])
    with pytest.raises(ValueError, match="cell_type_99"):
        model._cl_names_to_indices(cl_names_n, label_lookup)


def test_cl_names_to_indices_lookup_reused_across_batches():
    """The same label_lookup dict object is returned on repeated _get_subset_info calls."""
    model = _make_socam()
    _, _, _, lookup_1 = model._get_subset_info(["cell_type_1", "cell_type_3"])
    _, _, _, lookup_2 = model._get_subset_info(["cell_type_3", "cell_type_1"])
    assert lookup_1 is lookup_2  # no re-creation between batches
