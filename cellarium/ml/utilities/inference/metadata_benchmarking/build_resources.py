import json
import logging
import typing as t

import networkx as nx
import owlready2
from smart_open import open

logging.basicConfig(level=logging.INFO)

from cellarium.ml.utilities.inference.metadata_benchmarking import utils


def build_ontology_ancestor_dictionary(
    owl_graph: nx.DiGraph,
    owl_names: t.List[str],
    owl_names_to_idx_map: t.Dict[str, int]
) -> t.Dict[str, t.List[str]]:
    """
    Build cell ontology ancestor dictionary resource. This resource allows a fast indexing of each CL node. It
    contains all ancestors for each of the CL node. It allows to fetch all ancestors for any node with a runtime O(1).

    :param owl_graph: CL ontology graph
    :param owl_names: list of CL names
    :param owl_names_to_idx_map: map from CL name to index
    """
    logging.info("Building cell ontology ancestor dictionary...")
    owl_ancestors_dictionary = {}

    for owl_name in owl_names:
        current_ancestors = []
        orders = []

        for owl_ancestor_name in sorted(nx.ancestors(owl_graph, owl_name)):
            cell_order_idx = owl_names_to_idx_map[owl_ancestor_name]
            current_ancestors.append(owl_ancestor_name)
            orders.append(cell_order_idx)

        sorted_ancestors = [ancestor for _, ancestor in sorted(zip(orders, current_ancestors))]

        owl_ancestors_dictionary[owl_name] = sorted_ancestors

    return owl_ancestors_dictionary


def build_benchmarking_ontology_dictionary_resource(
    owl_graph: nx.DiGraph,
    owl_names: t.List[str],
    n_hops: int
) -> t.Dict[str, t.Any]:
    """
    Build an ontology ancestor and descendant dictionary resource.

    This resource provides fast indexing of each node up to a specified number of hops (`n_hops`).
    It includes ancestors and descendants for all nodes, along with hop level information.
    The hop level information contains:

        * Ancestor nodes for each hop, up to `n_hops` levels deep:
            - If `hop_1`, it includes immediate ancestors.
            - If `hop_2`, it includes ancestors of ancestors, and so on.

        * Descendant nodes for each hop, up to `n_hops` levels deep:
            - If `hop_1`, it includes immediate descendants.
            - If `hop_2`, it includes descendants of descendants, and so on.

        * Union of all ancestor and descendant nodes for each hop level.

    The resource allows retrieval of all ancestors and descendants for any node at any hop with O(1) runtime.

    Output Example
    --------------

    .. code-block:: python

        {
            "key1": "value1",
            "key2": "value2",
            "nested_key": {
                "subkey1": "subvalue1",
                "subkey2": "subvalue2"
            }
        }

    :param owl_graph: The ontology graph.
    :param owl_names: A list of node names.
    :param n_hops: The number of hops to consider.
    """
    ontology_resource_dict = {}

    for owl_name in owl_names:
        ontology_resource_dict[owl_name] = {}
        ontology_resource_dict[owl_name]["all_ancestors"] = utils.get_all_ancestors(owl_graph, node=owl_name)
        ontology_resource_dict[owl_name]["all_descendants"] = utils.get_all_descendants(owl_graph, node=owl_name)

        for top_n in range(n_hops):
            nodes = utils.get_n_level_ancestors(owl_graph, node=owl_name, n=top_n)
            hop_all_ancestors = set.union(
                *[utils.get_all_ancestors(owl_graph, node=node_cl_name) for node_cl_name in list(nodes)]
            )
            hop_all_descendants = set.union(
                *[utils.get_all_descendants(owl_graph, node=node_cl_name) for node_cl_name in list(nodes)]
            )

            ontology_resource_dict[owl_name][f"hop_{top_n}"] = {
                "nodes": nodes,
                "all_ancestors": hop_all_ancestors,
                "all_descendants": hop_all_descendants,
            }

    return ontology_resource_dict

