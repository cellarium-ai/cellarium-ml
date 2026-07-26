# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def calculate_precision(tp: float, fp: float) -> float:
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, set):
        return value
    return set(value)


def _calculate_tps_and_fps(
    query_scores: np.ndarray,
    ground_truth_cl_name: str,
    cell_type_ontology_term_ids: Sequence[str],
    co_resource: Mapping[str, Mapping[str, Any]],
    num_hops: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    hops = [co_resource[ground_truth_cl_name][f"hop_{i}"] for i in range(num_hops + 1)]
    true_positives = [0.0] * len(hops)
    false_positives = [0.0] * len(hops)
    max_descendants = [0.0] * len(hops)
    min_descendants = [0.0] * len(hops)

    for match_cl_name, match_score in zip(cell_type_ontology_term_ids, query_scores):
        match_score = float(match_score)
        match_co_data = co_resource.get(str(match_cl_name))
        if match_co_data is None:
            continue

        match_ancestors = _as_set(match_co_data.get("all_ancestors"))
        match_descendants = _as_set(match_co_data.get("all_descendants"))

        for hop_idx, hop in enumerate(hops):
            hop_nodes = _as_set(hop.get("nodes"))
            hop_all_descendants = _as_set(hop.get("all_descendants"))
            hop_all_ancestors = _as_set(hop.get("all_ancestors"))
            match_cl_name = str(match_cl_name)

            if match_cl_name in hop_nodes.intersection(match_ancestors):
                true_positives[hop_idx] = max(match_score, true_positives[hop_idx])
            elif (
                match_cl_name not in hop_all_descendants.union(hop_all_ancestors)
                and not match_descendants.intersection(hop_all_descendants)
            ):
                false_positives[hop_idx] = max(match_score, false_positives[hop_idx])
            elif match_cl_name in hop_all_descendants:
                max_descendants[hop_idx] = max(match_score, max_descendants[hop_idx])
                min_descendants[hop_idx] = min(match_score, min_descendants[hop_idx])

    return true_positives, false_positives, max_descendants, min_descendants


def calculate_hop_metrics_for_cell(
    query_cell_id: str,
    query_scores: np.ndarray,
    ground_truth_cl_name: str,
    cell_type_ontology_term_ids: Sequence[str],
    co_resource: Mapping[str, Mapping[str, Any]],
    num_hops: int = 4,
) -> dict[str, float | str | None]:
    query_cell_metrics: dict[str, float | str | None] = {
        "query_cell_id": query_cell_id,
        "ground_truth_cl_name": ground_truth_cl_name,
    }

    if ground_truth_cl_name not in co_resource:
        query_cell_metrics["detail"] = f"Couldn't find cell type {ground_truth_cl_name} in Cell Ontology resource"
        for hop_idx in range(num_hops + 1):
            for metric in (
                "sensitivity",
                "specificity",
                "f1_score",
                "fp",
                "max_descendant_score",
                "min_descendant_score",
            ):
                query_cell_metrics[f"hop_{hop_idx}_{metric}"] = None
        return query_cell_metrics

    true_positives, false_positives, max_descendants, min_descendants = _calculate_tps_and_fps(
        query_scores=query_scores,
        ground_truth_cl_name=ground_truth_cl_name,
        cell_type_ontology_term_ids=cell_type_ontology_term_ids,
        co_resource=co_resource,
        num_hops=num_hops,
    )

    sensitivities = true_positives
    specificities = [1 - fp for fp in false_positives]
    precisions = [calculate_precision(tp=tp, fp=fp) for tp, fp in zip(true_positives, false_positives)]
    f1_scores = [
        calculate_f1(precision=precision, recall=sensitivity)
        for precision, sensitivity in zip(precisions, sensitivities)
    ]

    for hop_idx, (sensitivity, specificity, f1_score, fp, max_descendant, min_descendant) in enumerate(
        zip(sensitivities, specificities, f1_scores, false_positives, max_descendants, min_descendants)
    ):
        query_cell_metrics[f"hop_{hop_idx}_sensitivity"] = sensitivity
        query_cell_metrics[f"hop_{hop_idx}_specificity"] = specificity
        query_cell_metrics[f"hop_{hop_idx}_f1_score"] = f1_score
        query_cell_metrics[f"hop_{hop_idx}_fp"] = fp
        query_cell_metrics[f"hop_{hop_idx}_max_descendant_score"] = max_descendant
        query_cell_metrics[f"hop_{hop_idx}_min_descendant_score"] = min_descendant

    return query_cell_metrics


def calculate_hop_metrics_for_batch(
    query_cell_ids: Sequence[str],
    prediction_scores_nc: np.ndarray,
    ground_truth_cl_names: Sequence[str],
    cell_type_ontology_term_ids: Sequence[str],
    co_resource: Mapping[str, Mapping[str, Any]],
    num_hops: int = 4,
) -> pd.DataFrame:
    if len(query_cell_ids) != len(ground_truth_cl_names):
        raise ValueError("`query_cell_ids` and `ground_truth_cl_names` must have the same length.")
    if prediction_scores_nc.shape[0] != len(query_cell_ids):
        raise ValueError("`prediction_scores_nc` must have one row per query cell.")
    if prediction_scores_nc.shape[1] != len(cell_type_ontology_term_ids):
        raise ValueError("`prediction_scores_nc` must have one column per cell ontology term.")

    rows = [
        calculate_hop_metrics_for_cell(
            query_cell_id=str(query_cell_id),
            query_scores=prediction_scores_nc[idx],
            ground_truth_cl_name=str(ground_truth_cl_name),
            cell_type_ontology_term_ids=cell_type_ontology_term_ids,
            co_resource=co_resource,
            num_hops=num_hops,
        )
        for idx, (query_cell_id, ground_truth_cl_name) in enumerate(zip(query_cell_ids, ground_truth_cl_names))
    ]
    return pd.DataFrame(rows)
