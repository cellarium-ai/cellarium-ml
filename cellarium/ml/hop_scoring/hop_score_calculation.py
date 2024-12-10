# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import concurrent.futures as concurrency
import logging
import multiprocessing
import traceback
import typing as t

import numpy as np
import pandas as pd

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

def calculate_tps_and_fps_csv(query_cell_obj: np.array, ground_truth_cl_name: str, cell_type_ontology_term_id_array: np.array, num_hops: int, co_resource: t.Dict[str, t.Any]
) -> t.Tuple[t.List[float], t.List[float]]:
    """
    Calculate true positives and false positives for each hop level.

    :param query_cell_obj: The query cell object containing ontology aware scores for.
    :param ground_truth_cl_name: The ground truth cell type name.
    :param num_hops: Number of hops to consider.
    :param co_resource: Cell Ontology helper resource which has already precalculated top_n_level descendants and
        ancestors.

    :return: A tuple of lists containing true positive and false positive scores for each hop level.
    """
    hops = [co_resource[ground_truth_cl_name][f"hop_{i}"] for i in range(num_hops + 1)]
    true_positives = [0.0] * len(hops)
    false_positives = [0.0] * len(hops)
    max_descendants = [0.0] * len(hops) #TRIAL
    min_descendants = [0.0] * len(hops) #TRIAL

    #for match in query_cell_obj.matches: # query_cell_obj["matches"]
    for i in range(len(query_cell_obj)):
        #match_cl_name = match.cell_type_ontology_term_id.replace("CL:","CL_")
        #match_cl_name = cell_type_ontology_term_id_array[i].replace("CL:","CL_")
        match_cl_name = cell_type_ontology_term_id_array[i]
        #match_score = match.score
        match_score = query_cell_obj[i]
        match_co_data = co_resource[match_cl_name]
        match_ancestors = match_co_data["all_ancestors"]

        for i, hop in enumerate(hops):
            hop_match_intersect = hop["nodes"].intersection(match_ancestors)
            hop_all_descendants = hop["all_descendants"]
            hop_all_ancestors = hop["all_ancestors"]

            if match_cl_name in hop_match_intersect:
                true_positives[i] = max(match_score, true_positives[i])
            elif match_cl_name not in hop_all_descendants.union(hop_all_ancestors):
                false_positives[i] = max(match_score, false_positives[i])
            elif match_cl_name in hop_all_descendants:
                max_descendants[i] = max(match_score,max_descendants[i]) #TRIAL
                min_descendants[i] = max(match_score,min_descendants[i]) #TRIAL

    return true_positives, false_positives, max_descendants, min_descendants

def calculate_metrics_for_query_cell_csv(
    db_id: int,
    query_cell_obj: np.array,
    ground_truth_cl_name: str,
    cell_type_ontology_term_id_array: np.array,
    co_resource: t.Dict[str, t.Any],
    num_hops=4,
):
    """
    Calculate performance metrics for a query cell against the ground truth.

    :param query_cell_obj: The query cell object containing ontology aware scores for.
    :param ground_truth_cl_name: The ground truth cell type label.
    :param co_resource: Cell ontology precalculated resource used in CZI.
    :param num_hops: Number of hops to consider.

    :return: A dictionary containing sensitivity, specificity, and F1 score for each hop level.
    """
    if ground_truth_cl_name not in co_resource:
        metrics_na = {
            "query_cell_id": query_cell_obj["query_cell_id"],
            "detail": f"Couldn't find cell type {ground_truth_cl_name} in Cell Ontology resource",
        }
        for hop in range(num_hops + 1):
            for metric in ["sensitivity", "specificity", "f1_score"]:
                metrics_na[f"hop_{hop}_{metric}"] = None

        return metrics_na

    true_positives, false_positives, max_descendants, min_descendants = calculate_tps_and_fps_csv(
        query_cell_obj=query_cell_obj,
        ground_truth_cl_name=ground_truth_cl_name,
        cell_type_ontology_term_id_array = cell_type_ontology_term_id_array,
        num_hops=num_hops,
        co_resource=co_resource,
    )

    sensitivities = [tp for tp in true_positives]
    specificities = [1 - fp for fp in false_positives]
    precisions = [calculate_precision(tp=tp, fp=fp) for tp, fp, in zip(true_positives, false_positives)]
    f1_scores = [
        calculate_f1(precision=precision, recall=sensitivity)
        for precision, sensitivity in zip(precisions, sensitivities)
    ]

    #query_cell_metrics = {"query_cell_id": query_cell_obj.query_cell_id, "detail": ""}
    query_cell_metrics = {"query_cell_id": db_id}

    # for i, (sensitivity, specificity, f1_score) in enumerate(zip(sensitivities, specificities, f1_scores)):
    #     query_cell_metrics[f"hop_{i}_sensitivity"] = sensitivity
    #     query_cell_metrics[f"hop_{i}_specificity"] = specificity
    #     query_cell_metrics[f"hop_{i}_f1_score"] = f1_score

    for i, (sensitivity, specificity, f1_score, fp, max_descendant, min_descendant) in enumerate(zip(sensitivities, specificities, f1_scores, false_positives, max_descendants, min_descendants)):
        query_cell_metrics[f"hop_{i}_sensitivity"] = sensitivity
        query_cell_metrics[f"hop_{i}_specificity"] = specificity
        query_cell_metrics[f"hop_{i}_f1_score"] = f1_score
        #query_cell_metrics[f"hop_{i}_tp"] = tp_i #Sensitivity is true positive so no need to save it twice
        query_cell_metrics[f"hop_{i}_fp"] = fp
        query_cell_metrics[f"hop_{i}_max_descendant_score"] = max_descendant
        query_cell_metrics[f"hop_{i}_min_descendant_score"] = min_descendant
    return query_cell_metrics

def calculate_metrics_for_csv_cas_output(
    db_id_array: np.array,
    ground_truth_cl_names: t.Iterable[str],
    cell_type_ontology_term_id_array: np.array,
    #cas_result: t.List[t.Dict[str, t.Any]],
    single_batch_df_copy: pd.DataFrame,
    co_resource: t.Dict[str, t.Any],
    num_hops: int,
) -> pd.DataFrame:
    """
    Calculate performance metrics for CAS (Cell Annotation Service) output.

    This function processes the CAS output and the ground truth cell types from the input AnnData object,
    computes performance metrics (sensitivity, specificity, and F1 score) for each hop level, and returns
    the results as a pandas DataFrame.

    :param ground_truth_cl_names: Iterable containing ground truths to benchmark against
    :param cas_result: The result from the CAS annotation, containing query results and matches for each cell.
    :param co_resource: Cell ontology precalculated resource used in CZI.
    :param num_hops: Number of hops to consider.

    :return: A pandas DataFrame with query cell IDs as the index and sensitivity, specificity, and F1 score
         for each hop level as columns.
    """
    df_result = pd.DataFrame()

    # for query_res_obj, ground_truth in zip(cas_result, ground_truth_cl_names):
    #     query_cell_metrics = calculate_metrics_for_query_cell(
    #         query_cell_obj=query_res_obj,
    #         ground_truth_cl_name=ground_truth,
    #         co_resource=co_resource,
    #         num_hops=num_hops,
    #     )

    #cell_type_ontology_term_id_array = np.array(single_batch_df_copy.columns)
    #print(f"NIMISH DBID ARRAY IS {db_id_array}")
    for i in range(len(single_batch_df_copy)):
        query_cell_metrics = calculate_metrics_for_query_cell_csv(
            db_id=db_id_array[i],
            query_cell_obj = single_batch_df_copy.iloc[i],
            ground_truth_cl_name = ground_truth_cl_names[i],
            cell_type_ontology_term_id_array = cell_type_ontology_term_id_array,
            co_resource = co_resource,
            num_hops = num_hops,
        )

        df_result = pd.concat([df_result, pd.DataFrame([query_cell_metrics])])

    #df_result = df_result.set_index("query_cell_id")
    return df_result

# Trial with cas metrics in batches
def split_into_batches_csv(data, batch_size):
    """
    Split a list into batches of a given size.

    :param data_list: List of data to be split into batches.
    :param batch_size: The size of each batch.

    :return: List of batches, where each batch is a list.
    """
    if type(data)==pd.DataFrame:
        return [data.iloc[i:i + batch_size] for i in range(0, len(data), batch_size)]
    else:
        return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

def calculate_metrics_for_cas_output_in_batches_csv(
    ground_truth_cl_names: t.List[str],
    #cas_result: t.Dict[str, t.Any],
    #co_resource: t.Dict[str, t.Any],
    db_id_array: np.array,
    cell_type_ontology_term_id_array: np.array,
    single_batch_df_copy: pd.DataFrame,
    co_resource: t.Dict[str, t.Any],
    num_hops: int,
    batch_size: int = 20000,
) -> pd.DataFrame:
    """
    Calculate metrics for CAS output in batches using multiprocessing.

    :param ground_truth_cl_names: List of ground truth labels.
    :param cas_result: List of CAS result dictionaries.
    :param co_resource: Cell ontology precalculated resource used in CZI.
    :param num_hops: Number of hops for evaluation.
    :param batch_size: Size of each batch. Default is 5000.

    :raises ValueError: If the length of ground_truths and cas_result do not match.

    :return: DataFrame containing the calculated metrics.
    """
    result_dfs = []
    num_workers = multiprocessing.cpu_count()
    with concurrency.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        #cas_result_batches = split_into_batches(data_list=cas_result, batch_size=batch_size)
        single_batch_df_copy_batches = split_into_batches_csv(data = single_batch_df_copy, batch_size=batch_size)
        #ground_truth_cl_names_batches = split_into_batches(data_list=ground_truth_cl_names, batch_size=batch_size)
        ground_truth_cl_names_batches = split_into_batches_csv(data=ground_truth_cl_names, batch_size=batch_size)
        db_id_array_batches = split_into_batches_csv(data = db_id_array, batch_size=batch_size)

        for i, (single_batch_df_copy_batch, ground_truth_cl_names_batch, db_id_array_batch) in enumerate(
            zip(single_batch_df_copy_batches, ground_truth_cl_names_batches, db_id_array_batches)
        ):
            calculate_metrics_kwargs = {
                "db_id_array": db_id_array_batch,
                "ground_truth_cl_names": ground_truth_cl_names_batch,
                "single_batch_df_copy": single_batch_df_copy_batch.reset_index(drop=True),
                "cell_type_ontology_term_id_array": cell_type_ontology_term_id_array,
                "co_resource": co_resource,
                "num_hops": num_hops,
            }
            logger.info(f"Submitting batch {i}. Length: {len(single_batch_df_copy_batch)}...")
            future = executor.submit(calculate_metrics_for_csv_cas_output, **calculate_metrics_kwargs)
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
