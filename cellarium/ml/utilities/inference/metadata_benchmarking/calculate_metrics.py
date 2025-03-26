import concurrent.futures as concurrency
import logging
import multiprocessing
import traceback
import typing as t

import numpy as np
import anndata
import pandas as pd
from smart_open import open

logger = logging.getLogger(__name__)


def calculate_precision(tp: float, fp: float) -> float:
    """
    Calculate precision.

    :param tp: True positives.
    :param fp: False positives.
    :return: Precision value.
    """
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score.

    :param precision: Precision value.
    :param recall: Recall value.

    :return: F1 score.
    """
    try:
        return (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return 0.0


def calculate_tps_and_fps(
    query_obj: t.Dict[str, t.Any],
    ground_truth_ontology_term_id: str,
    num_hops: int,
    ontology_resource: t.Dict[str, t.Any]
) -> t.Tuple[t.List[float], t.List[float]]:
    """
    Calculate true positives and false positives for each hop level.

    :param query_cell_obj: The query object containing ontology-aware scores.
    :param ground_truth_ontology_term_id: The ground truth ontology term ID.
    :param num_hops: Number of hops to consider.
    :param co_resource: Ontology helper resource which has already precalculated top_n_level descendants and
        ancestors.

    :return: A tuple of lists containing true positive and false positive scores for each hop level.
    """
    hops = [ontology_resource[ground_truth_ontology_term_id][f"hop_{i}"] for i in range(num_hops + 1)]
    true_positives = [0.0] * len(hops)
    false_positives = [0.0] * len(hops)

    for match in query_obj["matches"]:
        match_ontology_term_id = match["ontology_term_id"]
        match_score = match["score"]
        match_ontology_data = ontology_resource[match_ontology_term_id]
        match_ancestors = match_ontology_data["all_ancestors"]

        for i, hop in enumerate(hops):
            hop_match_intersect = hop["nodes"].intersection(match_ancestors)
            hop_all_descendants = hop["all_descendants"]
            hop_all_ancestors = hop["all_ancestors"]

            if match_ontology_term_id in hop_match_intersect:
                true_positives[i] = max(match_score, true_positives[i])
            elif match_ontology_term_id not in hop_all_descendants.union(hop_all_ancestors):
                false_positives[i] = max(match_score, false_positives[i])

    return true_positives, false_positives


def calculate_hop_inclusion_scores(
    query_obj: t.Dict[str, t.Any],
    ground_truth_ontology_term_id: str,
    num_hops: int,
    ontology_resource: t.Dict[str, t.Any]
) -> t.List[float]:
    """
    Calculate inclusion score at each hop level.

    :param query_cell_obj: The query object containing ontology-aware scores.
    :param ground_truth_ontology_term_id: The ground truth ontology term ID.
    :param num_hops: Number of hops to consider.
    :param co_resource: Ontology helper resource which has already precalculated top_n_level descendants and
        ancestors.

    :return: A list containing inclusion score at each hop level.
    """
    hop_ground_truth_ontology_contexts = [
        ontology_resource[ground_truth_ontology_term_id][f"hop_{i}"] for i in range(num_hops + 1)]
    inclusion_scores = [0] * len(hop_ground_truth_ontology_contexts)

    for i_hop, hop_ground_truth_ontology_context in enumerate(hop_ground_truth_ontology_contexts):
        for match in query_obj["matches"]:
            truth_set = hop_ground_truth_ontology_context["all_descendants"]
            if match["ontology_term_id"] in truth_set:
                inclusion_scores[i_hop] = inclusion_scores[i_hop] + match["score"]

    return inclusion_scores

def calculate_hop_calls(
    query_obj: t.Dict[str, t.Any],
    ground_truth_ontology_term_id: str,
    num_hops: int,
    ontology_resource: t.Dict[str, t.Any]
) -> t.List[float]:
    """
    Calculate correctness of argmax call at each hop level.

    :param query_cell_obj: The query object containing ontology-aware scores.
    :param ground_truth_ontology_term_id: The ground truth ontology term ID.
    :param num_hops: Number of hops to consider.
    :param co_resource: Ontology helper resource which has already precalculated top_n_level descendants and
        ancestors.

    :return: A list containing correctness of argmax call at each hop level.
    """
    hop_ground_truth_ontology_contexts = [
        ontology_resource[ground_truth_ontology_term_id][f"hop_{i}"] for i in range(num_hops + 1)]
    calls = [0] * len(hop_ground_truth_ontology_contexts)

    best_score = -np.inf
    best_ontology_term_id = None
    for match in query_obj["matches"]:
        if match["score"] > best_score:
            best_score = match["score"]
            best_ontology_term_id = match["ontology_term_id"]
    
    if best_ontology_term_id is None:
        raise RuntimeError("No best ontology term ID found -- this should not happen!") 
    
    for i_hop, hop_ground_truth_ontology_context in enumerate(hop_ground_truth_ontology_contexts):
        truth_set = hop_ground_truth_ontology_context["all_descendants"]
        if best_ontology_term_id in truth_set:
            calls[i_hop] = 1

    return calls


def calculate_metrics_for_query_v1(
    query_obj: t.Dict[str, t.Any],
    ground_truth_ontology_term_id: str,
    ontology_resource: t.Dict[str, t.Any],
    num_hops=4,
):
    """
    Calculate performance metrics for a query cell against the ground truth.

    :param query_obj: The query object containing ontology aware scores.
    :param ground_truth_ontology_term_id: The ground truth ontology term ID.
    :param ontology_resource: Ontology precalculated resource used in CELLxGENE.
    :param num_hops: Number of hops to consider.

    :return: A dictionary containing sensitivity, specificity, and F1 score for each hop level.
    """
    if ground_truth_ontology_term_id not in ontology_resource:
        metrics_na = {
            "query_cell_id": query_obj["query_cell_id"],
            "detail": f"Couldn't find {ground_truth_ontology_term_id} in the ontology resource",
        }
        for hop in range(num_hops + 1):
            for metric in ["sensitivity", "specificity", "f1_score"]:
                metrics_na[f"hop_{hop}_{metric}"] = np.nan

        return metrics_na

    true_positives, false_positives = calculate_tps_and_fps(
        query_obj=query_obj,
        ground_truth_ontology_term_id=ground_truth_ontology_term_id,
        num_hops=num_hops,
        ontology_resource=ontology_resource,
    )

    sensitivities = [tp for tp in true_positives]
    specificities = [1 - fp for fp in false_positives]
    precisions = [calculate_precision(tp=tp, fp=fp) for tp, fp, in zip(true_positives, false_positives)]
    f1_scores = [
        calculate_f1(precision=precision, recall=sensitivity)
        for precision, sensitivity in zip(precisions, sensitivities)
    ]

    query_cell_metrics = {"query_cell_id": query_obj["query_cell_id"], "detail": ""}

    for i, (sensitivity, specificity, f1_score) in enumerate(zip(sensitivities, specificities, f1_scores)):
        query_cell_metrics[f"hop_{i}_sensitivity"] = sensitivity
        query_cell_metrics[f"hop_{i}_specificity"] = specificity
        query_cell_metrics[f"hop_{i}_f1_score"] = f1_score

    return query_cell_metrics

def calculate_metrics_for_query_v2(
    query_obj: t.Dict[str, t.Any],
    ground_truth_ontology_term_id: str,
    ontology_resource: t.Dict[str, t.Any],
    num_hops=4,
):
    """
    Calculate performance metrics for a query cell against the ground truth.

    :param query_obj: The query object containing ontology aware scores.
    :param ground_truth_ontology_term_id: The ground truth ontology term ID.
    :param ontology_resource: Ontology precalculated resource used in CELLxGENE.
    :param num_hops: Number of hops to consider.

    :return: A dictionary containing sensitivity, specificity, and F1 score for each hop level.
    """
    if ground_truth_ontology_term_id not in ontology_resource:
        metrics_na = {
            "query_cell_id": query_obj["query_cell_id"],
            "detail": f"Couldn't find {ground_truth_ontology_term_id} in the ontology resource",
        }
        for hop in range(num_hops + 1):
                metrics_na[f"hop_{hop}_inclusion_score"] = np.nan

        return metrics_na

    hop_inclusion_scores = calculate_hop_inclusion_scores(
        query_obj=query_obj,
        ground_truth_ontology_term_id=ground_truth_ontology_term_id,
        num_hops=num_hops,
        ontology_resource=ontology_resource,
    )

    query_cell_metrics = {"query_cell_id": query_obj["query_cell_id"], "detail": ""}

    for i, hop_inclusion_score in enumerate(hop_inclusion_scores):
        query_cell_metrics[f"hop_{i}_inclusion_score"] = hop_inclusion_score

    return query_cell_metrics

def calculate_metrics_for_query_v3(
    query_obj: t.Dict[str, t.Any],
    ground_truth_ontology_term_id: str,
    ontology_resource: t.Dict[str, t.Any],
    num_hops=4,
):
    """
    Calculate performance metrics for a query cell against the ground truth.

    :param query_obj: The query object containing ontology aware scores.
    :param ground_truth_ontology_term_id: The ground truth ontology term ID.
    :param ontology_resource: Ontology precalculated resource used in CELLxGENE.
    :param num_hops: Number of hops to consider.

    :return: A dictionary containing sensitivity, specificity, and F1 score for each hop level.
    """
    if ground_truth_ontology_term_id not in ontology_resource:
        metrics_na = {
            "query_cell_id": query_obj["query_cell_id"],
            "detail": f"Couldn't find {ground_truth_ontology_term_id} in the ontology resource",
        }
        for hop in range(num_hops + 1):
                metrics_na[f"hop_{hop}_call"] = np.nan

        return metrics_na

    hop_calls = calculate_hop_calls(
        query_obj=query_obj,
        ground_truth_ontology_term_id=ground_truth_ontology_term_id,
        num_hops=num_hops,
        ontology_resource=ontology_resource,
    )

    query_cell_metrics = {"query_cell_id": query_obj["query_cell_id"], "detail": ""}

    for i, hop_call in enumerate(hop_calls):
        query_cell_metrics[f"hop_{i}_call"] = hop_call

    return query_cell_metrics


def calculate_metrics_for_prediction_output(
    ground_truth_ontology_term_ids: t.Iterable[str],
    model_predictions: t.List[t.Dict[str, t.Any]],
    ontology_resource: t.Dict[str, t.Any],
    num_hops: int,
    metric_style: t.Literal["hop_k_sensitivity_precision", "hop_k_inclusion", "hop_k_call"] = "hop_k_sensitivity_precision",
) -> pd.DataFrame:
    """
    Calculate performance metrics for metadata predictions.

    This function processes the predictions and ground truth terms from the input AnnData object,
    computes performance metrics (sensitivity, specificity, and F1 score) for each hop level, and returns
    the results as a pandas DataFrame.

    :param ground_truth_ontology_term_ids: Iterable containing ground truths to benchmark against
    :param model_predictions: The result from the metadata predictions, containing query results and matches for each cell.
    :param co_resource: Cell ontology precalculated resource used in CZI.
    :param num_hops: Number of hops to consider.

    :return: A pandas DataFrame with query cell IDs as the index and sensitivity, specificity, and F1 score
         for each hop level as columns.
    """
    metrics_list = []
    
    if metric_style == "hop_k_sensitivity_precision":
        metrics_calculator = calculate_metrics_for_query_v1
    elif metric_style == "hop_k_inclusion":
        metrics_calculator = calculate_metrics_for_query_v2
    elif metric_style == "hop_k_call":
        metrics_calculator = calculate_metrics_for_query_v3
    else:
        raise ValueError(f"Unknown algorithm: {metric_style}")
    
    for query_res_obj, ground_truth in zip(model_predictions, ground_truth_ontology_term_ids):
        query_cell_metrics = metrics_calculator(
            query_obj=query_res_obj,
            ground_truth_ontology_term_id=ground_truth,
            ontology_resource=ontology_resource,
            num_hops=num_hops,
        )
        metrics_list.append(query_cell_metrics)

    df_result = pd.DataFrame(metrics_list).set_index("query_cell_id")
    return df_result


def split_into_batches(data_list, batch_size):
    """
    Split a list into batches of a given size.

    :param data_list: List of data to be split into batches.
    :param batch_size: The size of each batch.

    :return: List of batches, where each batch is a list.
    """
    return [data_list[i : i + batch_size] for i in range(0, len(data_list), batch_size)]


def calculate_metrics_for_cas_output_in_batches(
    ground_truth_ontology_term_ids: t.List[str],
    model_predictions: t.Dict[str, t.Any],
    ontology_resource: t.Dict[str, t.Any],
    num_hops: int,
    batch_size: int = 20000,
) -> pd.DataFrame:
    """
    Calculate performance metrics in batches using multiprocessing.

    :param ground_truth_ontology_term_ids: List of ground truth labels.
    :param model_predictions: List of prediction result dictionaries.
    :param ontology_resource: Ontology precalculated resource used in CELLxGENE.
    :param num_hops: Number of hops for evaluation.
    :param batch_size: Size of each batch. Default is 5000.

    :raises ValueError: If the length of ground_truth_ontology_term_ids and cas_result do model_predictions match.

    :return: DataFrame containing the calculated metrics.
    """
    result_dfs = []
    num_workers = multiprocessing.cpu_count()
    with concurrency.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        model_predictions_batches = split_into_batches(
            data_list=model_predictions["data"], batch_size=batch_size)
        ground_truth_ontology_term_ids_batches = split_into_batches(
            data_list=ground_truth_ontology_term_ids, batch_size=batch_size)

        for i, (model_predictions_batch, ground_truth_ontology_term_ids_batch) in enumerate(
            zip(model_predictions_batches, ground_truth_ontology_term_ids_batches)
        ):
            calculate_metrics_kwargs = {
                "model_predictions": model_predictions_batch,
                "ground_truth_ontology_term_ids": ground_truth_ontology_term_ids_batch,
                "ontology_resource": ontology_resource,
                "num_hops": num_hops,
            }
            logger.info(f"Submitting batch {i}. Length: {len(model_predictions_batch)}...")
            future = executor.submit(calculate_metrics_for_prediction_output, **calculate_metrics_kwargs)
            futures.append(future)

        logger.info("Waiting for metrics to be calculated...")
        done, not_done = concurrency.wait(futures, return_when=concurrency.ALL_COMPLETED)
        logger.info("Aggregating results")
        for future in done:
            try:
                # Attempt to get the result of the future
                batch_metrics_df = future.result()
            except Exception as e:
                # If an exception is raised, print the exception details
                logging.error(f"Exception occurred: {e}")
                # Format and print the full traceback
                traceback.print_exception(type(e), e, e.__traceback__)
            else:
                result_dfs.append(batch_metrics_df)

    df_result = pd.concat(result_dfs)
    logger.info("Done.")
    return df_result


