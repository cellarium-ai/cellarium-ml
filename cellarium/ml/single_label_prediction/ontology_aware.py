from collections import defaultdict
from operator import itemgetter

# AnnData-related constants
CAS_CL_SCORES_ANNDATA_OBSM_KEY = "socam_cl_scores"
CAS_METADATA_ANNDATA_UNS_KEY = "socam_metadata"
# the 'cell' node
CL_CELL_ROOT_NODE = "CL_0000000"

def get_most_granular_top_k_calls(
    aggregated_scores_c,
    cl,
    cl_names,
    min_acceptable_score: float,
    top_k: int = 1,
    root_note: str = CL_CELL_ROOT_NODE,
    
):
    depth_list = list(map(cl.get_longest_path_lengths_from_target(root_note).get, cl_names))
    sorted_score_and_depth_list = sorted(
        list(
            (score, depth, cl_name)
            for score, depth, cl_name in zip(
                aggregated_scores_c, depth_list, cl_names
            )
            if score >= min_acceptable_score
        ),
        key=itemgetter(1),
        reverse=True,
    )
    trunc_list = sorted_score_and_depth_list[:top_k]
    # pad with root node if necessary
    for _ in range(top_k - len(trunc_list)):
        trunc_list.append((1.0, 0, root_note))
    return trunc_list

def compute_most_granular_top_k_calls_single(
    adata,
    cl,
    cl_names,
    cl_names_to_labels_map,
    min_acceptable_score: float,
    top_k: int = 3,
    obs_prefix: str = "cas_cell_type",
    root_note: str = CL_CELL_ROOT_NODE,
):
    top_k_calls_dict = defaultdict(list)
    scores_array_nc = adata.obsm[CAS_CL_SCORES_ANNDATA_OBSM_KEY]

    # make a wrapped aggregated scores dataclass for the single cell in question

    for i_cell in range(adata.n_obs):
        aggregated_scores_c = scores_array_nc[i_cell]
        top_k_output = get_most_granular_top_k_calls(
            aggregated_scores_c, 
            cl,
            cl_names, 
            min_acceptable_score, 
            top_k, 
            root_note)
        for k in range(top_k):
            top_k_calls_dict[f"{obs_prefix}_score_{k + 1}"].append(top_k_output[k][0])
            top_k_calls_dict[f"{obs_prefix}_name_{k + 1}"].append(top_k_output[k][2])
            top_k_calls_dict[f"{obs_prefix}_label_{k + 1}"].append(cl_names_to_labels_map[top_k_output[k][2]])

    for k, v in top_k_calls_dict.items():
        adata.granular_results[k] = v