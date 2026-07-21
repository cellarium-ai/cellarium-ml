# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""scANVI: Single-cell Annotation using Variational Inference."""

import warnings
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from cellarium.ml.layers import DressedLayer, FullyConnectedLinear
from cellarium.ml.models.scvi import (
    SingleCellVariationalInference,
    compute_annealed_kl_weight,
    weights_init,
)
from cellarium.ml.models.socam import (
    _build_nonleaf_info,
    _expand_with_ancestors,
    _propagate_logits_impl,
    _propagate_probs_impl,
    compute_class_weights,
    propagate_logits,
    propagate_probs,
)
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

_EPS = 1e-8


class ConditionalNormalEncoder(torch.nn.Module):
    """Encode concatenated [z, one_hot(c)] to a Normal distribution.

    Expects 2D input ``[N, in_features]``. The caller is responsible for reshaping
    3D inputs ``[N, chunk, in_features]`` to ``[N*chunk, in_features]`` before calling
    and reshaping the output back.

    LayerNorm (rather than BatchNorm) is used in the hidden layers so that the module is robust
    to the very small, variable batch sizes produced by per-label grouped marginalization (a
    singleton group yields a batch of size 1, which BatchNorm cannot handle in train mode).

    Args:
        in_features: Input dimensionality (typically ``n_latent + n_partition``).
        out_features: Output (latent) dimensionality.
        n_hidden: Widths of hidden layers. Empty list gives a single linear layer.
        use_layer_norm: Whether to apply ``LayerNorm`` in hidden layers.
        dropout_rate: Dropout rate in hidden layers.
        var_eps: Floor added to the variance for numerical stability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_hidden: list[int],
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.var_eps = var_eps

        trunk_layers: list[torch.nn.Module] = []
        current_in = in_features
        for h in n_hidden:
            trunk_layers.append(
                DressedLayer(
                    torch.nn.Linear(current_in, h),
                    use_layer_norm=use_layer_norm,
                    dropout_rate=dropout_rate,
                )
            )
            current_in = h

        self.trunk = torch.nn.Sequential(*trunk_layers) if trunk_layers else torch.nn.Identity()
        self.mean_head = torch.nn.Linear(current_in, out_features, bias=True)
        self.log_var_head = torch.nn.Linear(current_in, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> Normal:
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_var = self.log_var_head(h)
        return Normal(mean, (log_var.exp() + self.var_eps).sqrt())


def compute_frontier(
    descendant_tensor: torch.Tensor,
    cl_names: list[str],
    counts: torch.Tensor,
    min_cells: int,
    excluded_names: set[str],
) -> tuple[list[str], list[str]]:
    """Compute a support-backed frontier antichain through a cell ontology.

    The frontier is the set of deepest nodes whose subtree cell-count (own + all descendant
    counts) is at least ``min_cells``, with a coverage-repair pass that guarantees every
    *observed* node maps to exactly one frontier ancestor. Repair adds the shallowest ancestor
    of an orphaned node that is not itself an ancestor of any already-chosen frontier node (see
    the module docstring / design discussion), so under-supported isolated branches become their
    own small frontier partitions rather than being dropped.

    Args:
        descendant_tensor: ``(C, C)`` binary tensor, diagonal 1, where entry ``[i, j] == 1``
            means ``j`` is a descendant of ``i`` (``i`` is an ancestor of ``j``).
        cl_names: Ordered node names matching rows/columns of ``descendant_tensor``.
        counts: ``(C,)`` per-node *direct* cell counts aligned to ``cl_names``.
        min_cells: Support threshold for the cut.
        excluded_names: Node names never eligible for the frontier (e.g. the unlabeled sentinel).

    Returns:
        ``(frontier_names, under_supported_names)`` — the frontier node names (a subset of
        ``cl_names``, in ``cl_names`` order) and the subset of those that fell below
        ``min_cells`` (added by coverage repair).
    """
    # Pin construction-time math to CPU so it works inside a torch.device("meta") context
    # (the passed-in descendant_tensor/counts are concrete CPU tensors).
    desc = descendant_tensor.float().cpu()
    n = len(cl_names)
    excluded_mask = torch.tensor([name in excluded_names for name in cl_names], dtype=torch.bool, device="cpu")

    counts = counts.clone().float().cpu()
    counts[excluded_mask] = 0.0
    subtree_count = desc @ counts  # [C], row i sums over descendants of i (incl. self)

    qualifies = (subtree_count >= min_cells) & (~excluded_mask) & (subtree_count > 0)
    proper_desc = desc.clone()
    proper_desc.fill_diagonal_(0.0)
    has_qualifying_proper_desc = (proper_desc @ qualifies.float()) > 0
    frontier_mask = qualifies & (~has_qualifying_proper_desc)

    observed = (counts > 0) & (~excluded_mask)

    def _covered(mask: torch.Tensor) -> torch.Tensor:
        # A node w is covered if some frontier node is an ancestor of w (finer-than-frontier,
        # binned up: desc[f, w] == 1) OR a descendant of w (coarse, marginalized: desc[w, f] == 1).
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return torch.zeros(n, dtype=torch.bool, device="cpu")
        has_frontier_ancestor = desc[idx, :].sum(dim=0) > 0
        has_frontier_descendant = desc[:, idx].sum(dim=1) > 0
        return has_frontier_ancestor | has_frontier_descendant

    covered = _covered(frontier_mask)
    orphans = (observed & (~covered)).nonzero(as_tuple=True)[0].tolist()
    for w in orphans:
        if covered[w]:
            continue  # already handled by a previous repair addition
        # allowed ancestors of w: a is an ancestor of w, not excluded, and not an ancestor
        # of any current frontier node
        is_ancestor_of_w = desc[:, w] > 0
        frontier_idx = frontier_mask.nonzero(as_tuple=True)[0]
        is_ancestor_of_frontier = (
            desc[:, frontier_idx].sum(dim=1) > 0
            if frontier_idx.numel() > 0
            else torch.zeros(n, dtype=torch.bool, device="cpu")
        )
        allowed = is_ancestor_of_w & (~is_ancestor_of_frontier) & (~excluded_mask)
        allowed_idx = allowed.nonzero(as_tuple=True)[0]
        # shallowest allowed ancestor = the one with the most descendants
        descendant_counts = desc[allowed_idx, :].sum(dim=1)
        chosen = allowed_idx[int(torch.argmax(descendant_counts).item())]
        frontier_mask[chosen] = True
        covered = _covered(frontier_mask)

    frontier_indices = frontier_mask.nonzero(as_tuple=True)[0].tolist()
    frontier_names = [cl_names[i] for i in frontier_indices]
    under_supported = [cl_names[i] for i in frontier_indices if subtree_count[i].item() < min_cells]
    return frontier_names, under_supported


class SCANVI(SingleCellVariationalInference):
    r"""Single-cell ANnotation using Variational Inference (scANVI) [1].

    Extends scVI with a hierarchical prior on the latent variable ``z``:

    .. math::

        c \sim p(c), \quad u \sim \mathcal{N}(0, I),
        \quad z \sim p(z \mid u, c), \quad x \sim p(x \mid z, s)

    The encoder ``q(z|x)`` is identical to scVI. Three new networks are added:

    * **Classifier** ``q(c|z)``: predicts cell-type probabilities from ``z``.
    * **u-encoder** ``q(u|z,c)``: encodes a deeper latent ``u`` given ``z`` and cell type ``c``.
    * **z-prior decoder** ``p(z|u,c)``: provides the class-conditional prior on ``z``.

    The class variable ``c`` ranges over a **partition** of cell-type space. Two classifier
    modes are supported:

    * ``classifier_type="flat"`` (default): a plain MLP over a flat list of cell types
      (``cell_type_categories``), with standard cross-entropy — the classic scANVI behaviour.
    * ``classifier_type="ontology"``: a SOCAM-style ontology-aware MLP that emits logits over
      all active ontology nodes (leaf and internal) and propagates them up the Cell Ontology.
      The generative partition is a **support-backed frontier** (see :func:`compute_frontier`)
      computed from ``class_counts`` at construction. A coarse (internal-node) label licenses a
      restricted marginalization of the ELBO over the frontier leaves in its subtree, so labels
      at any granularity supervise the model. Optional inverse-frequency class balancing is
      applied to the (hierarchical) cross-entropy term.

    For labeled cells the ELBO is maximized using the known cell type (or, for a coarse label,
    marginalized over its subtree) together with an auxiliary cross-entropy classification term.
    For unlabeled cells (label equal to ``unlabeled_category``) the cell-type expectation is
    marginalized over the whole partition in chunks to control peak VRAM.

    The classifier is **batch-agnostic** (it consumes only ``z``), so with a batch-agnostic
    encoder configuration one can go ``x -> z -> c`` at inference time.

    **References:**

    1. `Probabilistic harmonization and annotation of single-cell transcriptomics data
       with deep generative models (Xu et al., 2021)
       <https://doi.org/10.15252/msb.20209620>`_.

    Args:
        classifier_type:
            ``"flat"`` (default) for a plain MLP over ``cell_type_categories``; ``"ontology"``
            for the SOCAM-style ontology-aware classifier.
        classification_weight:
            Weight ``α`` applied to the cross-entropy loss on labeled cells.
        chunk_size:
            Number of partition members processed per chunk during marginalization. Reduce this
            to lower peak VRAM at the cost of slightly more computation.
        classifier_n_hidden:
            Hidden layer widths for the classifier ``q(c|z)`` MLP. Defaults to ``[128]``.
        classifier_dropout_rate:
            Dropout rate in classifier hidden layers.
        secondary_n_hidden:
            Hidden layer widths shared by the u-encoder ``q(u|z,c)`` and the z-prior decoder
            ``p(z|u,c)``. Defaults to ``[128]``.
        y_prior_probs:
            Prior probabilities over the partition ``p(c)``. Must sum to 1 and have length equal
            to the partition size. If ``None`` a uniform prior is used (recommended).
        unlabeled_category:
            String label marking unlabeled cells. Default ``"unknown"``.
        descendant_tensor:
            (ontology mode) ``(C, C)`` binary tensor, diagonal 1, where ``[i, j] == 1`` means
            ``j`` is a descendant of ``i``. Rows/columns are indexed by ``cl_names``.
        cl_names:
            (ontology mode) Ordered node names matching ``descendant_tensor``.
        class_counts:
            (ontology mode) pandas Series mapping node names to training cell counts. Drives both
            the frontier cut and (optionally) the class-balancing weights.
        frontier_min_cells:
            (ontology mode) Support threshold for the frontier cut.
        propagate_class_counts:
            (ontology mode) If True, propagate counts up the ontology before computing balancing
            weights (see :func:`~cellarium.ml.models.socam.compute_class_weights`).
        probability_propagation_flag:
            (ontology mode) If True, apply hierarchical propagation to the classifier logits.
        use_torch_compile:
            (ontology mode) If True, use the ``torch.compile``d propagation functions. Defaults
            to False (eager) to avoid recompilation across variable batch shapes.
        marginalization_warn_fraction:
            (ontology mode) Warn (once per label) when a coarse label marginalizes over more than
            this fraction of the frontier — a signal that the label is nearly as expensive as
            unlabeled while providing little supervision.
        **scvi_kwargs:
            All keyword arguments accepted by
            :class:`~cellarium.ml.models.SingleCellVariationalInference`. In flat mode,
            ``cell_type_categories`` defines the partition and is required.
    """

    def __init__(
        self,
        classifier_type: Literal["flat", "ontology"] = "flat",
        classification_weight: float = 50.0,
        chunk_size: int = 100,
        classifier_n_hidden: list[int] | None = None,
        classifier_dropout_rate: float = 0.1,
        secondary_n_hidden: list[int] | None = None,
        y_prior_probs: list[float] | None = None,
        unlabeled_category: str = "unknown",
        descendant_tensor: torch.Tensor | None = None,
        cl_names: list[str] | None = None,
        class_counts: pd.Series | None = None,
        frontier_min_cells: int = 50,
        propagate_class_counts: bool = False,
        probability_propagation_flag: bool = True,
        use_torch_compile: bool = False,
        marginalization_warn_fraction: float = 0.25,
        **scvi_kwargs,
    ):
        if scvi_kwargs.get("use_flow", False):
            raise ValueError("SCANVI does not support use_flow=True.")

        super().__init__(**scvi_kwargs)

        if classifier_n_hidden is None:
            classifier_n_hidden = [128]
        if secondary_n_hidden is None:
            secondary_n_hidden = [128]

        self.classifier_type = classifier_type
        self.classification_weight = classification_weight
        self.chunk_size = chunk_size
        self.unlabeled_category = unlabeled_category
        self.probability_propagation_flag = probability_propagation_flag
        self.use_torch_compile = use_torch_compile
        self.marginalization_warn_fraction = marginalization_warn_fraction
        self._warned_marg_labels: set[str] = set()

        if classifier_type == "flat":
            self._init_flat()
        elif classifier_type == "ontology":
            self._init_ontology(
                descendant_tensor=descendant_tensor,
                cl_names=cl_names,
                class_counts=class_counts,
                frontier_min_cells=frontier_min_cells,
                propagate_class_counts=propagate_class_counts,
            )
        else:
            raise ValueError(f"classifier_type must be 'flat' or 'ontology', got {classifier_type!r}.")

        # Classifier q(c|z): MLP from latent space to logits over the classifier output space.
        self.cell_type_classifier = FullyConnectedLinear(
            in_features=self.n_latent,
            out_features=self._classifier_out_dim,
            n_hidden=classifier_n_hidden,
            dressing_init_kwargs={"use_batch_norm": True, "dropout_rate": classifier_dropout_rate},
            bias=True,
        )

        # q(u|z,c): encodes the deeper latent u given z and (one-hot) partition membership c
        self.u_encoder = ConditionalNormalEncoder(
            in_features=self.n_latent + self.n_partition,
            out_features=self.n_latent,
            n_hidden=secondary_n_hidden,
        )
        # p(z|u,c): class-conditional prior on z, parameterized by u and partition membership c
        self.z_prior_decoder = ConditionalNormalEncoder(
            in_features=self.n_latent + self.n_partition,
            out_features=self.n_latent,
            n_hidden=secondary_n_hidden,
        )

        # Prior over the partition p(c); uniform by default (kept as numpy so reset_parameters()
        # can restore the buffer after meta-device materialization wipes it).
        if y_prior_probs is not None:
            if len(y_prior_probs) != self.n_partition:
                raise ValueError(
                    f"len(y_prior_probs)={len(y_prior_probs)} does not match partition size {self.n_partition}."
                )
            y_prior_np = np.asarray(y_prior_probs, dtype=np.float32)
            y_prior_np = y_prior_np / y_prior_np.sum()
        else:
            y_prior_np = np.ones(self.n_partition, dtype=np.float32) / self.n_partition
        self._y_prior_numpy = y_prior_np
        self.register_buffer("y_prior", torch.as_tensor(y_prior_np))

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Mode-specific construction
    # ------------------------------------------------------------------

    def _init_flat(self) -> None:
        """Build the flat-mode partition from ``cell_type_categories``."""
        if self.cell_type_categories is None:
            raise ValueError("classifier_type='flat' requires cell_type_categories to define the class partition.")
        self.class_names: list[str] = list(self.cell_type_categories)
        self.n_partition = len(self.class_names)
        self._classifier_out_dim = self.n_partition
        self._label_to_partition_idx: dict[str, int] = {name: i for i, name in enumerate(self.class_names)}

    def _init_ontology(
        self,
        descendant_tensor: torch.Tensor | None,
        cl_names: list[str] | None,
        class_counts: pd.Series | None,
        frontier_min_cells: int,
        propagate_class_counts: bool,
    ) -> None:
        """Build the ontology-mode active set, frontier partition, and class weights."""
        if descendant_tensor is None or cl_names is None:
            raise ValueError("classifier_type='ontology' requires descendant_tensor and cl_names.")
        # Pin to CPU so construction works inside a torch.device("meta") context.
        descendant_tensor = descendant_tensor.float().cpu()
        if descendant_tensor.shape[0] != descendant_tensor.shape[1]:
            raise ValueError("descendant_tensor must be square.")
        if descendant_tensor.trace().item() != descendant_tensor.shape[0]:
            raise ValueError("descendant_tensor must have ones on the diagonal (each node is its own descendant).")
        if len(cl_names) != descendant_tensor.shape[0]:
            raise ValueError("len(cl_names) must match descendant_tensor.shape[0].")
        if class_counts is None:
            raise ValueError("classifier_type='ontology' requires class_counts to compute the frontier.")

        self.cl_names = list(cl_names)
        cl_index = {name: i for i, name in enumerate(self.cl_names)}

        # Direct counts aligned to cl_names (missing -> 0).
        direct_counts = torch.zeros(len(self.cl_names), dtype=torch.float, device="cpu")
        for name, cnt in class_counts.items():
            if name in cl_index:
                if cnt < 0:
                    raise ValueError("All class_counts values must be >= 0.")
                direct_counts[cl_index[name]] = float(cnt)

        frontier_names, under_supported = compute_frontier(
            descendant_tensor=descendant_tensor,
            cl_names=self.cl_names,
            counts=direct_counts,
            min_cells=frontier_min_cells,
            excluded_names={self.unlabeled_category},
        )
        if len(frontier_names) == 0:
            raise ValueError("Computed frontier is empty; check class_counts and frontier_min_cells.")
        if under_supported:
            warnings.warn(
                f"{len(under_supported)} frontier node(s) fall below frontier_min_cells={frontier_min_cells} "
                f"and were added by coverage repair to avoid orphaning cells: {under_supported}",
                UserWarning,
            )

        # Active set = frontier plus all their ancestors (SOCAM-style).
        active_cl_names = _expand_with_ancestors(frontier_names, self.cl_names, descendant_tensor)
        self.active_cl_names = active_cl_names
        self.n_active = len(active_cl_names)
        active_index = {name: i for i, name in enumerate(active_cl_names)}
        ix = torch.tensor([cl_index[name] for name in active_cl_names], dtype=torch.long, device="cpu")
        active_desc = descendant_tensor[ix][:, ix]  # (n_active, n_active)
        nonleaf_info = _build_nonleaf_info(active_desc)

        # Active leaves == frontier == the generative partition.
        leaf_mask = active_desc.sum(dim=1) == 1
        frontier_active_idx = leaf_mask.nonzero(as_tuple=True)[0]  # (n_frontier,) in active-column space
        self.n_partition = int(frontier_active_idx.numel())
        self._classifier_out_dim = self.n_active
        self.frontier_cl_names = [active_cl_names[i] for i in frontier_active_idx.tolist()]

        # frontier_membership_af[a, p] == 1 iff frontier partition member p is a descendant of active node a
        frontier_membership_af = active_desc[:, frontier_active_idx]  # (n_active, n_frontier)

        # Class-balancing weights over all active nodes (data-frequency mean-1 for reduction="none").
        class_weights = compute_class_weights(
            active_cl_names=active_cl_names,
            class_counts=class_counts,
            active_descendant_tensor_cc=active_desc,
            propagate_class_counts=propagate_class_counts,
            normalize="data_mean",
        )

        # Label resolver: every cl_name -> active node index (finer-than-frontier binned to its
        # frontier ancestor; coarse labels map to their own active node).
        self._label_to_active_idx: dict[str, int] = {}
        for name in self.cl_names:
            if name == self.unlabeled_category:
                continue
            if name in active_index:
                self._label_to_active_idx[name] = active_index[name]
            else:
                # finer than the frontier: bin up to the unique frontier ancestor
                j = cl_index[name]
                for fname in self.frontier_cl_names:
                    if descendant_tensor[cl_index[fname], j] > 0:
                        self._label_to_active_idx[name] = active_index[fname]
                        break

        # CPU copies for reset_parameters() restore after meta-device materialization.
        self._active_descendant_tensor_cc = active_desc
        self._nonleaf_desc_cc = nonleaf_info["nonleaf_desc_cc"]
        self._perm = nonleaf_info["perm"]
        self._inv_perm = nonleaf_info["inv_perm"]
        self._frontier_active_idx = frontier_active_idx
        self._frontier_membership_af = frontier_membership_af
        self._class_weights = class_weights

        self.register_buffer("active_descendant_tensor_cc", active_desc.clone())
        self.register_buffer("nonleaf_desc_cc", nonleaf_info["nonleaf_desc_cc"].clone())
        self.register_buffer("perm", nonleaf_info["perm"].clone())
        self.register_buffer("inv_perm", nonleaf_info["inv_perm"].clone())
        self.register_buffer("frontier_active_idx", frontier_active_idx.clone())
        self.register_buffer("frontier_membership_af", frontier_membership_af.clone())
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone())
        else:
            self.register_buffer("class_weights", None)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        # SingleCellVariationalInference.__init__ calls reset_parameters() before the
        # SCANVI-specific attributes are built; skip on that first call.
        if not hasattr(self, "cell_type_classifier"):
            return
        self.cell_type_classifier.apply(weights_init)
        self.u_encoder.apply(weights_init)
        self.z_prior_decoder.apply(weights_init)
        with torch.no_grad():
            self.y_prior.copy_(torch.as_tensor(self._y_prior_numpy, device=self.y_prior.device))
            if self.classifier_type == "ontology":
                self.active_descendant_tensor_cc.copy_(self._active_descendant_tensor_cc)
                self.nonleaf_desc_cc.copy_(self._nonleaf_desc_cc)
                self.perm.copy_(self._perm)
                self.inv_perm.copy_(self._inv_perm)
                self.frontier_active_idx.copy_(self._frontier_active_idx)
                self.frontier_membership_af.copy_(self._frontier_membership_af)
                if self._class_weights is not None:
                    self.class_weights.copy_(self._class_weights)

    # ------------------------------------------------------------------
    # Classifier heads (mode-specific)
    # ------------------------------------------------------------------

    def _partition_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Return ``q(c|z)`` over the partition, shape ``[N, n_partition]``."""
        if self.classifier_type == "flat":
            return F.softmax(logits, dim=-1)
        return F.softmax(logits[:, self.frontier_active_idx], dim=-1)

    def _propagate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        fn = propagate_logits if self.use_torch_compile else _propagate_logits_impl
        return fn(logits, self.nonleaf_desc_cc, self.perm, self.inv_perm)

    def _propagate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        fn = propagate_probs if self.use_torch_compile else _propagate_probs_impl
        return fn(probs, self.active_descendant_tensor_cc)

    def _supervised_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-cell (hierarchical, class-weighted) cross-entropy, shape ``[m]``."""
        if self.classifier_type == "flat":
            return F.cross_entropy(logits, target, reduction="none")
        propagated = logits
        if self.probability_propagation_flag:
            propagated = self._propagate_logits(logits)
        return F.cross_entropy(propagated, target, reduction="none", weight=self.class_weights)

    # ------------------------------------------------------------------
    # Marginalization
    # ------------------------------------------------------------------

    def _conditional_kls(
        self,
        z: torch.Tensor,
        qz_mean: torch.Tensor,
        qz_std: torch.Tensor,
        c_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``KL(q(z|x) || p(z|u,c))`` and ``KL(q(u|z,c) || N(0,I))`` for 2D inputs.

        Args:
            z: Latent sample ``[N, n_latent]``.
            qz_mean: Mean of ``q(z|x)`` ``[N, n_latent]``.
            qz_std: Std of ``q(z|x)`` ``[N, n_latent]``.
            c_onehot: One-hot partition membership ``[N, n_partition]``.

        Returns:
            ``(kl_z, kl_u)`` each ``[N]``.
        """
        zc = torch.cat([z, c_onehot], dim=-1)
        qu = self.u_encoder(zc)
        u = qu.rsample()
        uc = torch.cat([u, c_onehot], dim=-1)
        pz = self.z_prior_decoder(uc)
        qz = Normal(qz_mean, qz_std)
        pu = Normal(torch.zeros_like(u), torch.ones_like(u))
        kl_z = torch.distributions.kl_divergence(qz, pz).sum(dim=-1)
        kl_u = torch.distributions.kl_divergence(qu, pu).sum(dim=-1)
        return kl_z, kl_u

    def _marginalize_over_set(
        self,
        z_m: torch.Tensor,
        qz_mean_m: torch.Tensor,
        qz_std_m: torch.Tensor,
        q_partition_m: torch.Tensor,
        set_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Marginalize ``kl_z``/``kl_u`` over a partition subset and compute the restricted ``kl_c``.

        Args:
            z_m: Latent samples for the group ``[m, n_latent]``.
            qz_mean_m: ``q(z|x)`` mean ``[m, n_latent]``.
            qz_std_m: ``q(z|x)`` std ``[m, n_latent]``.
            q_partition_m: Full partition posterior for the group ``[m, n_partition]``.
            set_idx: Partition indices ``S`` to marginalize over ``[s]``.

        Returns:
            ``(kl_z, kl_u, kl_c)`` each ``[m]``. ``kl_c`` is exactly 0 when ``|S| == 1``.
        """
        m = z_m.shape[0]
        s = int(set_idx.numel())
        device = z_m.device

        q_S = q_partition_m[:, set_idx]  # [m, s]
        q_S_norm = q_S / q_S.sum(dim=-1, keepdim=True).clamp_min(_EPS)  # [m, s]

        # Restricted kl_c = KL(q(.|z, c in S) || p(.|c in S))
        p_S = self.y_prior[set_idx]
        p_S = p_S / p_S.sum().clamp_min(_EPS)
        kl_c = (q_S_norm * (q_S_norm.clamp_min(_EPS).log() - p_S.unsqueeze(0).clamp_min(_EPS).log())).sum(dim=-1)

        kl_z = torch.zeros(m, device=device)
        kl_u = torch.zeros(m, device=device)
        for start in range(0, s, self.chunk_size):
            end = min(start + self.chunk_size, s)
            chunk = end - start
            members = set_idx[start:end]  # [chunk]
            c_onehot = F.one_hot(members, self.n_partition).float()  # [chunk, n_partition]

            z_exp = z_m.unsqueeze(1).expand(-1, chunk, -1).reshape(m * chunk, -1)
            qzm_exp = qz_mean_m.unsqueeze(1).expand(-1, chunk, -1).reshape(m * chunk, -1)
            qzs_exp = qz_std_m.unsqueeze(1).expand(-1, chunk, -1).reshape(m * chunk, -1)
            c_exp = c_onehot.unsqueeze(0).expand(m, -1, -1).reshape(m * chunk, -1)

            kl_z_c, kl_u_c = self._conditional_kls(z_exp, qzm_exp, qzs_exp, c_exp)
            kl_z_c = kl_z_c.view(m, chunk)
            kl_u_c = kl_u_c.view(m, chunk)
            w = q_S_norm[:, start:end]
            kl_z = kl_z + (w * kl_z_c).sum(dim=-1)
            kl_u = kl_u + (w * kl_u_c).sum(dim=-1)

        return kl_z, kl_u, kl_c

    def _resolve_group(self, label: str) -> tuple[torch.Tensor, int | None]:
        """Resolve a label string to ``(marg_partition_idx, ce_target_or_None)``.

        ``ce_target`` indexes the classifier output space (partition idx in flat mode, active
        node idx in ontology mode). It is ``None`` for the unlabeled category.
        """
        all_frontier = torch.arange(self.n_partition, dtype=torch.long)
        if label == self.unlabeled_category:
            return all_frontier, None
        if self.classifier_type == "flat":
            if label not in self._label_to_partition_idx:
                raise ValueError(f"Unknown cell type label {label!r} (not in cell_type_categories).")
            idx = self._label_to_partition_idx[label]
            return torch.tensor([idx], dtype=torch.long), idx
        if label not in self._label_to_active_idx:
            raise ValueError(f"Cell type label {label!r} does not map to any active ontology node.")
        active_idx = self._label_to_active_idx[label]
        set_idx = (self.frontier_membership_af[active_idx] > 0).nonzero(as_tuple=True)[0].cpu()
        return set_idx, active_idx

    # ------------------------------------------------------------------
    # Vectorized label preprocessing and per-bucket KL helpers
    # ------------------------------------------------------------------

    def _preprocess_labels(
        self,
        labels: np.ndarray,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Partition cells into leaf, coarse, and unlabeled buckets (CPU preprocessing).

        Returns a dict with keys::

            leaf_idx, leaf_c_class, leaf_ce_target,
            coarse_idx, coarse_set_idx_padded, coarse_mask, coarse_ce_target,
            unlabeled_idx

        *leaf*: labeled cells whose resolved partition set has exactly one member.
        *coarse*: labeled cells whose resolved partition set has two or more members.
        *unlabeled*: cells whose label equals ``unlabeled_category``.
        """
        leaf_groups: list[tuple[np.ndarray, int, int]] = []
        coarse_groups: list[tuple[np.ndarray, torch.Tensor, int]] = []
        unlabeled_ids: list[int] = []

        for label in np.unique(labels):
            label_str = str(label)
            group = np.where(labels == label)[0]
            set_idx, ce_target = self._resolve_group(label_str)

            if label_str == self.unlabeled_category:
                unlabeled_ids.extend(group.tolist())
            else:
                assert ce_target is not None
                if set_idx.numel() == 1:
                    leaf_groups.append((group, int(set_idx[0].item()), ce_target))
                else:
                    if self.classifier_type == "ontology":
                        frac = set_idx.numel() / max(self.n_partition, 1)
                        if frac > self.marginalization_warn_fraction and label_str not in self._warned_marg_labels:
                            self._warned_marg_labels.add(label_str)
                            warnings.warn(
                                f"Coarse label {label_str!r} marginalizes over "
                                f"{set_idx.numel()}/{self.n_partition} frontier nodes "
                                f"({frac:.0%}); this costs nearly as much as an unlabeled cell.",
                                UserWarning,
                            )
                    coarse_groups.append((group, set_idx, ce_target))

        _empty = torch.empty(0, dtype=torch.long, device=device)

        if leaf_groups:
            leaf_cell_ids = np.concatenate([g for g, _, _ in leaf_groups])
            leaf_c_class_arr = np.concatenate([[c] * len(g) for g, c, _ in leaf_groups])
            leaf_ce_target_arr = np.concatenate([[t] * len(g) for g, _, t in leaf_groups])
            leaf_idx = torch.tensor(leaf_cell_ids, dtype=torch.long, device=device)
            leaf_c_class = torch.tensor(leaf_c_class_arr, dtype=torch.long, device=device)
            leaf_ce_target = torch.tensor(leaf_ce_target_arr, dtype=torch.long, device=device)
        else:
            leaf_idx = leaf_c_class = leaf_ce_target = _empty

        if coarse_groups:
            max_s = max(s.numel() for _, s, _ in coarse_groups)
            row_ids, row_set, row_mask, row_ce = [], [], [], []
            for grp, set_idx_k, ce_t in coarse_groups:
                n_k = len(grp)
                s = set_idx_k.numel()
                row_ids.append(grp)
                padded = np.zeros(max_s, dtype=np.int64)
                padded[:s] = set_idx_k.cpu().numpy()
                mask_row = np.zeros(max_s, dtype=bool)
                mask_row[:s] = True
                row_set.append(np.tile(padded, (n_k, 1)))
                row_mask.append(np.tile(mask_row, (n_k, 1)))
                row_ce.append(np.full(n_k, ce_t, dtype=np.int64))
            coarse_idx = torch.tensor(np.concatenate(row_ids), dtype=torch.long, device=device)
            coarse_set_idx_padded = torch.tensor(np.concatenate(row_set, axis=0), dtype=torch.long, device=device)
            coarse_mask = torch.tensor(np.concatenate(row_mask, axis=0), dtype=torch.bool, device=device)
            coarse_ce_target = torch.tensor(np.concatenate(row_ce), dtype=torch.long, device=device)
        else:
            coarse_idx = coarse_ce_target = _empty
            coarse_set_idx_padded = torch.empty((0, 0), dtype=torch.long, device=device)
            coarse_mask = torch.empty((0, 0), dtype=torch.bool, device=device)

        unlabeled_idx = torch.tensor(unlabeled_ids, dtype=torch.long, device=device) if unlabeled_ids else _empty

        return {
            "leaf_idx": leaf_idx,
            "leaf_c_class": leaf_c_class,
            "leaf_ce_target": leaf_ce_target,
            "coarse_idx": coarse_idx,
            "coarse_set_idx_padded": coarse_set_idx_padded,
            "coarse_mask": coarse_mask,
            "coarse_ce_target": coarse_ce_target,
            "unlabeled_idx": unlabeled_idx,
        }

    def _leaf_kls(
        self,
        z: torch.Tensor,
        qz_mean: torch.Tensor,
        qz_std: torch.Tensor,
        leaf_idx: torch.Tensor,
        leaf_c_class: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single vectorized KL pass for all singleton-set (leaf) cells.

        ``kl_c`` is identically 0 for leaf cells and is not returned.
        """
        c_onehot = F.one_hot(leaf_c_class, self.n_partition).float()  # [N_leaf, n_partition]
        return self._conditional_kls(z[leaf_idx], qz_mean[leaf_idx], qz_std[leaf_idx], c_onehot)

    def _coarse_kls(
        self,
        z: torch.Tensor,
        qz_mean: torch.Tensor,
        qz_std: torch.Tensor,
        q_partition: torch.Tensor,
        coarse_idx: torch.Tensor,
        coarse_set_idx_padded: torch.Tensor,
        coarse_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padded batched marginalization for all coarse-labeled cells in one GPU pass.

        All coarse cells are stacked with their frontier subsets padded to ``max_s``, so a single
        ``_conditional_kls`` call handles every cell simultaneously. The existing ``chunk_size``
        is applied along the padded-set dimension to bound peak VRAM.

        Returns:
            ``(kl_z, kl_u, kl_c)`` each ``[N_coarse]``.
        """
        N_coarse = coarse_idx.numel()
        max_s = coarse_set_idx_padded.shape[1]
        device = z.device

        # q(c|z) restricted to the subtree S, renormalized; padding positions -> 0.
        q_S = q_partition[coarse_idx].gather(1, coarse_set_idx_padded)  # [N_coarse, max_s]
        q_S = q_S * coarse_mask.float()
        q_S_norm = q_S / q_S.sum(dim=-1, keepdim=True).clamp_min(_EPS)  # [N_coarse, max_s]

        # Restricted kl_c (pure tensor arithmetic, no encoder call).
        # masked_fill keeps log finite at padding positions; clamp_min guards near-zero valid probs.
        p_S = self.y_prior[coarse_set_idx_padded]  # [N_coarse, max_s]
        p_S = p_S * coarse_mask.float()
        p_S_norm = p_S / p_S.sum(dim=-1, keepdim=True).clamp_min(_EPS)

        q_log = q_S_norm.masked_fill(~coarse_mask, 1.0).clamp_min(_EPS).log()
        p_log = p_S_norm.masked_fill(~coarse_mask, 1.0).clamp_min(_EPS).log()
        kl_c = (q_S_norm * (q_log - p_log)).sum(dim=-1)  # [N_coarse]

        # kl_z and kl_u: chunk over the padded-set dimension, all N_coarse cells in parallel.
        z_c = z[coarse_idx]  # [N_coarse, n_latent]
        qzm_c = qz_mean[coarse_idx]  # [N_coarse, n_latent]
        qzs_c = qz_std[coarse_idx]  # [N_coarse, n_latent]

        kl_z = torch.zeros(N_coarse, device=device)
        kl_u = torch.zeros(N_coarse, device=device)

        for start in range(0, max_s, self.chunk_size):
            end = min(start + self.chunk_size, max_s)
            chunk = end - start
            members = coarse_set_idx_padded[:, start:end]  # [N_coarse, chunk]
            w = q_S_norm[:, start:end] * coarse_mask[:, start:end].float()  # [N_coarse, chunk]

            c_onehot = F.one_hot(members, self.n_partition).float()  # [N_coarse, chunk, n_partition]
            z_exp = z_c.unsqueeze(1).expand(-1, chunk, -1).reshape(N_coarse * chunk, -1)
            qzm_exp = qzm_c.unsqueeze(1).expand(-1, chunk, -1).reshape(N_coarse * chunk, -1)
            qzs_exp = qzs_c.unsqueeze(1).expand(-1, chunk, -1).reshape(N_coarse * chunk, -1)
            c_exp = c_onehot.reshape(N_coarse * chunk, -1)

            kl_z_chunk, kl_u_chunk = self._conditional_kls(z_exp, qzm_exp, qzs_exp, c_exp)
            kl_z += (w * kl_z_chunk.view(N_coarse, chunk)).sum(dim=-1)
            kl_u += (w * kl_u_chunk.view(N_coarse, chunk)).sum(dim=-1)

        return kl_z, kl_u, kl_c

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,
        total_mrna_umis_n: torch.Tensor | None = None,
        cell_type_labels_n: np.ndarray | None = None,
    ) -> dict:
        """Compute the scANVI ELBO and auxiliary classification loss.

        Args:
            x_ng: Gene counts matrix ``[N, G]``.
            var_names_g: Variable names for input validation.
            batch_index_n: Integer batch indices ``[N]``.
            continuous_covariates_nc: Continuous covariates ``[N, C]``.
            categorical_covariate_index_nd: Integer categorical covariate codes ``[N, D]``.
            total_mrna_umis_n: Total mRNA UMIs per cell (not log-scaled).
            cell_type_labels_n: Per-cell string labels ``[N]``. Cells equal to
                ``unlabeled_category`` (default ``"unknown"``) are treated as unlabeled. Finer-than-
                frontier labels are binned to their frontier ancestor; coarse labels marginalize
                over their subtree. If ``None``, all cells are treated as unlabeled.

        Returns:
            A dict with ``loss``, ``reconstruction_loss``, ``kl_divergence_z``, ``kl_divergence_u``,
            ``kl_divergence_c``, ``kl_divergence_batch``, ``classification_loss``, ``z_nk``, and
            ``cell_type_logits_nc`` (classifier logits in its output space).
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        n = x_ng.shape[0]
        device = x_ng.device

        if cell_type_labels_n is None:
            labels = np.array([self.unlabeled_category] * n, dtype=object)
        else:
            labels = np.asarray(cell_type_labels_n).reshape(-1)

        batch_nb = self.batch_representation_from_batch_index(batch_index_n)
        categorical_covariate_np = self.categorical_onehot_from_categorical_index(categorical_covariate_index_nd)

        if self.use_size_factor_key:
            assert total_mrna_umis_n is not None, "total_mrna_umis_n required when use_size_factor_key=True"
            library_size_n1 = torch.log(total_mrna_umis_n).unsqueeze(-1)
        else:
            library_size_n1 = torch.log(x_ng.sum(dim=-1, keepdim=True))

        if self.input_gene_dropout_rate > 0.0:
            with torch.no_grad():
                dropout_mask_ng = torch.rand_like(x_ng) > self.input_gene_dropout_rate
                inference_input_x_ng = x_ng * dropout_mask_ng
        else:
            inference_input_x_ng = x_ng

        inference_outputs = self.inference(
            x_ng=inference_input_x_ng,
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
        )
        generative_outputs = self.generative(
            z_nk=inference_outputs["z"],
            library_size_n1=library_size_n1,
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
        )

        z = inference_outputs["z"]
        qz = inference_outputs["qz"]
        qz_mean = qz.mean
        qz_std = qz.stddev

        rec_loss_n = -generative_outputs["px"].log_prob(x_ng).sum(dim=-1)

        kl_annealing_weight = compute_annealed_kl_weight(
            epoch=self.epoch,
            step=self.step,
            n_epochs_kl_warmup=self.kl_warmup_epochs,
            n_steps_kl_warmup=self.kl_warmup_steps,
            max_kl_weight=1.0,
            min_kl_weight=self.kl_annealing_start,
        )

        kl_divergence_batch_n = torch.zeros(n, device=device)
        if self.batch_representation_sampled and (self.batch_kl_weight_max > 0):
            kl_divergence_batch_n = torch.distributions.kl_divergence(
                self.batch_embedding_distribution(batch_index_n=batch_index_n),
                Normal(torch.zeros_like(batch_nb), torch.ones_like(batch_nb)),
            ).sum(dim=1)

        # Classifier logits (batch-agnostic: depends only on z via qz.mean).
        logits = self.cell_type_classifier(qz_mean)  # [n, classifier_out_dim]
        q_partition = self._partition_probs(logits)  # [n, n_partition]

        kl_z_n = torch.zeros(n, device=device)
        kl_u_n = torch.zeros(n, device=device)
        kl_c_n = torch.zeros(n, device=device)
        ce_loss_n = torch.zeros(n, device=device)

        # Partition cells into leaf / coarse / unlabeled buckets, then process each in one GPU pass.
        buckets = self._preprocess_labels(labels, device)

        # Leaf path: all singleton-set cells in a single encoder call; kl_c == 0 identically.
        if buckets["leaf_idx"].numel() > 0:
            leaf_idx = buckets["leaf_idx"]
            kl_z_leaf, kl_u_leaf = self._leaf_kls(z, qz_mean, qz_std, leaf_idx, buckets["leaf_c_class"])
            kl_z_n[leaf_idx] = kl_z_leaf
            kl_u_n[leaf_idx] = kl_u_leaf
            ce_loss_n[leaf_idx] = self._supervised_ce(logits[leaf_idx], buckets["leaf_ce_target"])

        # Coarse path: all multi-set labeled cells in one padded encoder call.
        if buckets["coarse_idx"].numel() > 0:
            coarse_idx = buckets["coarse_idx"]
            kl_z_c, kl_u_c, kl_c_c = self._coarse_kls(
                z,
                qz_mean,
                qz_std,
                q_partition,
                coarse_idx,
                buckets["coarse_set_idx_padded"],
                buckets["coarse_mask"],
            )
            kl_z_n[coarse_idx] = kl_z_c
            kl_u_n[coarse_idx] = kl_u_c
            kl_c_n[coarse_idx] = kl_c_c
            ce_loss_n[coarse_idx] = self._supervised_ce(logits[coarse_idx], buckets["coarse_ce_target"])

        # Unlabeled path: single batch marginalized over the whole frontier with a chunk loop.
        if buckets["unlabeled_idx"].numel() > 0:
            unlabeled_idx = buckets["unlabeled_idx"]
            all_frontier = torch.arange(self.n_partition, dtype=torch.long, device=device)
            kl_z_u, kl_u_u, kl_c_u = self._marginalize_over_set(
                z[unlabeled_idx],
                qz_mean[unlabeled_idx],
                qz_std[unlabeled_idx],
                q_partition[unlabeled_idx],
                all_frontier,
            )
            kl_z_n[unlabeled_idx] = kl_z_u
            kl_u_n[unlabeled_idx] = kl_u_u
            kl_c_n[unlabeled_idx] = kl_c_u

        loss = torch.mean(
            rec_loss_n
            + kl_annealing_weight
            * (self.z_kl_weight_max * (kl_z_n + kl_u_n) + self.batch_kl_weight_max * kl_divergence_batch_n)
            + kl_c_n
            + self.classification_weight * ce_loss_n,
        )

        return {
            "loss": loss,
            "reconstruction_loss": rec_loss_n,
            "kl_divergence_z": kl_z_n,
            "kl_divergence_u": kl_u_n,
            "kl_divergence_c": kl_c_n,
            "kl_divergence_batch": kl_divergence_batch_n,
            "classification_loss": ce_loss_n,
            "z_nk": z,
            "cell_type_logits_nc": logits,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,
        total_mrna_umis_n: torch.Tensor | None = None,
        cell_type_labels_n: np.ndarray | None = None,
    ) -> None:
        """Log ``val_loss`` (ELBO) and a SOCAM-style cell-type accuracy on labeled cells.

        The accuracy is the mean propagated probability at the resolved ground-truth node
        (ontology mode) or the mean predicted probability of the true class (flat mode).
        """
        output = self(
            x_ng=x_ng,
            var_names_g=var_names_g,
            batch_index_n=batch_index_n,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_index_nd=categorical_covariate_index_nd,
            total_mrna_umis_n=total_mrna_umis_n,
            cell_type_labels_n=cell_type_labels_n,
        )
        loss = output["loss"]
        if isinstance(loss, torch.Tensor):
            pl_module.log("val_loss", loss, sync_dist=True, on_epoch=True, batch_size=x_ng.shape[0])

        if cell_type_labels_n is None:
            return
        labels = np.asarray(cell_type_labels_n).reshape(-1)
        labeled = labels != self.unlabeled_category
        if not labeled.any():
            return

        logits = output["cell_type_logits_nc"]
        assert isinstance(logits, torch.Tensor)

        if self.classifier_type == "flat":
            probs = F.softmax(logits, dim=-1)
            scores = []
            for i in np.where(labeled)[0]:
                label = str(labels[i])
                if label not in self._label_to_partition_idx:
                    continue
                scores.append(probs[i, self._label_to_partition_idx[label]])
        else:
            probs = self._propagate_probs(F.softmax(logits, dim=-1))
            scores = []
            for i in np.where(labeled)[0]:
                label = str(labels[i])
                if label not in self._label_to_active_idx:
                    continue
                scores.append(probs[i, self._label_to_active_idx[label]])

        if scores:
            metric = torch.stack(scores).mean()
            name = "val_ontology_accuracy" if self.classifier_type == "ontology" else "val_accuracy"
            pl_module.log(name, metric, sync_dist=True, on_epoch=True, batch_size=len(scores))

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,
    ) -> dict:
        """Embed cells and predict cell-type probabilities.

        If :attr:`reconstruct_counts_on_predict` is ``True``, falls back to the parent's
        count-reconstruction behaviour and omits cell-type probabilities.

        Args:
            x_ng: Gene counts matrix ``[N, G]``.
            var_names_g: Variable names for input validation.
            batch_index_n: Integer batch indices ``[N]``. (Consumed by the encoder; may be a dummy
                when the encoder is configured batch-agnostic.)
            continuous_covariates_nc: Continuous covariates ``[N, C]``.
            categorical_covariate_index_nd: Integer categorical covariate codes ``[N, D]``.

        Returns:
            A dict with ``x_ng`` (latent embeddings ``[N, n_latent]``) and ``cell_type_probs_nc``.
            In ontology mode ``cell_type_probs_nc`` is the propagated probability over all active
            nodes (``[N, n_active]``, columns = :attr:`active_cl_names`); in flat mode it is the
            softmax over the partition (``[N, n_partition]``).
        """
        if self.reconstruct_counts_on_predict:
            return super().predict(
                x_ng=x_ng,
                var_names_g=var_names_g,
                batch_index_n=batch_index_n,
                continuous_covariates_nc=continuous_covariates_nc,
                categorical_covariate_index_nd=categorical_covariate_index_nd,
            )

        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        batch_nb = self.batch_representation_from_batch_index(batch_index_n)
        categorical_covariate_np = self.categorical_onehot_from_categorical_index(categorical_covariate_index_nd)

        inference_outputs = self.inference(
            x_ng=x_ng,
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
        )
        qz = inference_outputs["qz"]
        z_nk = self._latent_value_from_latent_distribution(qz)
        logits = self.cell_type_classifier(qz.mean)
        if self.classifier_type == "flat":
            probs_nc = F.softmax(logits, dim=-1)
        else:
            probs_nc = self._propagate_probs(F.softmax(logits, dim=-1))

        return {
            "x_ng": z_nk,
            "cell_type_probs_nc": probs_nc,
        }
