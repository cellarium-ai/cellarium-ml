# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cellarium.ml.callbacks import SOCAMHopScoringPredictionWriter
from cellarium.ml.hop_scoring import calculate_hop_metrics_for_batch


def _hop(nodes: set[str], all_descendants: set[str], all_ancestors: set[str]) -> dict[str, set[str]]:
    return {
        "nodes": nodes,
        "all_descendants": all_descendants,
        "all_ancestors": all_ancestors,
    }


def _co_resource() -> dict[str, dict[str, object]]:
    root_hops = {
        f"hop_{idx}": _hop(nodes={"CL_root"}, all_descendants={"CL_root", "CL_child"}, all_ancestors={"CL_root"})
        for idx in range(5)
    }
    child_hops = {
        f"hop_{idx}": _hop(nodes={"CL_child"}, all_descendants={"CL_child"}, all_ancestors={"CL_root", "CL_child"})
        for idx in range(5)
    }
    other_hops = {
        f"hop_{idx}": _hop(nodes={"CL_other"}, all_descendants={"CL_other"}, all_ancestors={"CL_other"})
        for idx in range(5)
    }
    return {
        "CL_root": {
            "all_ancestors": {"CL_root"},
            "all_descendants": {"CL_root", "CL_child"},
            **root_hops,
        },
        "CL_child": {
            "all_ancestors": {"CL_root", "CL_child"},
            "all_descendants": {"CL_child"},
            **child_hops,
        },
        "CL_other": {
            "all_ancestors": {"CL_other"},
            "all_descendants": {"CL_other"},
            **other_hops,
        },
    }


def test_calculate_hop_metrics_for_batch():
    metrics_df = calculate_hop_metrics_for_batch(
        query_cell_ids=["cell_0"],
        prediction_scores_nc=np.array([[0.8, 0.2, 0.1]]),
        ground_truth_cl_names=["CL_root"],
        cell_type_ontology_term_ids=["CL_root", "CL_child", "CL_other"],
        co_resource=_co_resource(),
        num_hops=4,
    )

    row = metrics_df.iloc[0]
    assert row["query_cell_id"] == "cell_0"
    assert row["ground_truth_cl_name"] == "CL_root"
    assert row["hop_0_sensitivity"] == 0.8
    assert row["hop_0_specificity"] == 0.9
    assert row["hop_0_f1_score"] == pytest.approx(2 * 0.8 * (0.8 / 0.9) / (0.8 + (0.8 / 0.9)))
    assert row["hop_0_max_descendant_score"] == 0.2


def test_socam_hop_scoring_prediction_writer_writes_csv(tmp_path: Path):
    co_resource_path = tmp_path / "co_resource.pkl"
    pd.to_pickle(_co_resource(), co_resource_path)

    class DummyTrainer:
        world_size = 1
        global_rank = 0

    class DummyModel:
        active_cl_names = ["CL:root", "CL:child", "CL:other"]

    class DummyModule:
        model = DummyModel()

    writer = SOCAMHopScoringPredictionWriter(
        output_dir=tmp_path / "predictions",
        co_resource_path=str(co_resource_path),
    )
    writer.write_on_batch_end(
        trainer=DummyTrainer(),
        pl_module=DummyModule(),
        prediction={"cell_type_probs_nc": torch.tensor([[0.8, 0.2, 0.1]])},
        batch_indices=None,
        batch={
            "obs_names_n": np.array(["cell_0"]),
            "cl_names_n": np.array(["CL:root"]),
        },
        batch_idx=0,
        dataloader_idx=0,
    )

    output_df = pd.read_csv(tmp_path / "predictions" / "batch_0.csv")
    assert output_df.loc[0, "query_cell_id"] == "cell_0"
    assert output_df.loc[0, "ground_truth_cl_name"] == "CL_root"
    assert output_df.loc[0, "hop_4_sensitivity"] == pytest.approx(0.8)
