import typing as t
from functools import lru_cache

import networkx as nx
import owlready2
from scipy import sparse as sp

DEFAULT_CL_OWL_PATH = "https://github.com/obophenotype/cell-ontology/releases/download/v2024-01-04/cl.owl"
# DEFAULT_CL_OWL_PATH = "/home/nmagre/P1_logistic_regression/cell_ontology_ancestors/cl.owl"

# only keep nodes with the following prefix when parsing CL ontology
CL_PREFIX = "CL_"

# the 'cell' node
CL_CELL_ROOT_NODE = "CL_0000000"

# the 'eukaryotic cell' node
CL_EUKARYOTIC_CELL_ROOT_NODE = "CL_0000255"


class CellOntologyCache:
    """
    A class representing a cache for the Cell Ontology (CL).

    Args:
        cl_owl_path (str): The path to the Cell Ontology OWL file. Defaults to DEFAULT_CL_OWL_PATH.

    Attributes:
        cl_classes (list): A list of Cell Ontology classes with a singleton label.
        cl_names (list): A list of names of Cell Ontology classes.
        cl_labels (list): A list of labels of Cell Ontology classes.
        cl_classes_set (set): A set of Cell Ontology classes.
        cl_names_to_labels_map (dict): A mapping of Cell Ontology class names to labels.
        cl_labels_to_names_map (dict): A mapping of Cell Ontology class labels to names.
        cl_names_to_idx_map (dict): A mapping of Cell Ontology class names to indices.
        cl_idx_to_names_map (dict): A mapping of indices to Cell Ontology class names.
        cl_graph (nx.DiGraph): A networkx graph representing the Cell Ontology.

    """

    def __init__(self, cl_owl_path: str = DEFAULT_CL_OWL_PATH):
        """
        Initialize the CellOntologyCache object.

        Loads the Cell Ontology from the specified OWL file, filters out classes with a singleton label,
        and builds a networkx graph representing the Cell Ontology.

        :param cl_owl_path: The path to the Cell Ontology OWL file. Defaults to DEFAULT_CL_OWL_PATH.
        :type cl_owl_path: str

        """


        cl = owlready2.get_ontology(cl_owl_path).load()

        # only keep CL classes with a singleton label
        cl_classes = list(
            _class for _class in cl.classes() if _class.name.startswith(CL_PREFIX) and len(_class.label) == 1
        )

        cl_names = [_class.name for _class in cl_classes]
        cl_labels = [_class.label[0] for _class in cl_classes]
        assert len(set(cl_names)) == len(cl_classes)
        assert len(set(cl_labels)) == len(cl_classes)

        cl_classes_set = set(cl_classes)
        cl_names_to_labels_map = {cl_name: cl_label for cl_name, cl_label in zip(cl_names, cl_labels)}
        cl_labels_to_names_map = {cl_label: cl_name for cl_name, cl_label in zip(cl_names, cl_labels)}
        cl_names_to_idx_map = {cl_name: idx for idx, cl_name in enumerate(cl_names)}
        cl_idx_to_names_map = {idx: cl_name for idx, cl_name in enumerate(cl_names)}

        # build a networkx graph from CL
        cl_graph = nx.DiGraph(name="CL graph")

        for cl_class in cl_classes:
            cl_graph.add_node(cl_class.name)

        for self_cl_class in cl_classes:
            for parent_cl_class in cl.get_parents_of(self_cl_class):
                if parent_cl_class not in cl_classes_set:
                    continue
                cl_graph.add_edge(parent_cl_class.name, self_cl_class.name)
            for child_cl_class in cl.get_children_of(self_cl_class):
                if child_cl_class not in cl_classes_set:
                    continue
                cl_graph.add_edge(self_cl_class.name, child_cl_class.name)

        self.cl_classes = cl_classes
        self.cl_names = cl_names
        self.cl_labels = cl_labels
        self.cl_classes_set = cl_classes_set
        self.cl_names_to_labels_map = cl_names_to_labels_map
        self.cl_labels_to_names_map = cl_labels_to_names_map
        self.cl_names_to_idx_map = cl_names_to_idx_map
        self.cl_idx_to_names_map = cl_idx_to_names_map
        self.cl_graph = cl_graph

    @property
    @lru_cache(maxsize=None)
    def cl_ancestors_csr_matrix(self) -> sp.csr_matrix:
        """Returns a sparse matrix representation of ancestors.

        .. note:
            The matrix element (i, j) = 1 iff j is an ancetor of i.
        """
        n_nodes = len(self.cl_graph.nodes)

        row = []
        col = []
        data = []

        for cl_name in self.cl_names:
            self_idx = self.cl_names_to_idx_map[cl_name]
            for cl_ancestor_name in nx.ancestors(self.cl_graph, cl_name):
                ancestor_idx = self.cl_names_to_idx_map[cl_ancestor_name]
                row.append(self_idx)
                col.append(ancestor_idx)
                data.append(1)

        cl_ancestors_csr_matrix = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        return cl_ancestors_csr_matrix

    @lru_cache(maxsize=None)
    def get_longest_path_lengths_from_target(self, target: str) -> t.Dict[str, float]:
        # Perform a topological sort of the graph
        topo_order = list(nx.topological_sort(self.cl_graph))

        # Initialize distances with -infinity for all nodes except the target
        distances = {node: float("-inf") for node in self.cl_graph.nodes()}
        distances[target] = 0

        # Process nodes in topological order
        for node in topo_order:
            if distances[node] != float("-inf"):  # Only process reachable nodes
                for neighbor in self.cl_graph.successors(node):
                    if distances[neighbor] < distances[node] + 1:
                        distances[neighbor] = distances[node] + 1

        return distances
