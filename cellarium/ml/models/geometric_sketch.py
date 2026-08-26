# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class StreamingGeometricSketch(CellariumModel):
    """
    Online geometric sketching via locality-sensitive hashing (LSH).

    Streams single-cell gene expression data and retains a geometrically diverse
    sketch of cells across a single pass. Rare cell states are preserved while
    over-represented dense clusters are capped at ``max_cells_per_bucket``.

    Cells are bucketed by projecting into a low-dimensional space and binarizing
    the result to produce an integer bucket ID. Each bucket maintains a reservoir
    of up to ``max_cells_per_bucket`` cells via uniform reservoir sampling.

    By default the projection is a frozen random Gaussian linear map from gene
    space to ``n_bits`` dimensions (classical random-hyperplane LSH). Optionally
    a pre-trained module (e.g. a frozen PCA or scVI encoder) can be provided as
    ``projector``; its output is then passed through the LSH random linear layer,
    so ``n_bits`` controls bucket granularity independently of the encoder's output
    dimension.

    Only single-device training is supported. A ``RuntimeError`` is raised at the
    start of training if more than one device is detected.

    Args:
        var_names_g:
            Gene names for input validation.
        n_bits:
            Number of LSH projection bits. Determines up to ``2^n_bits`` buckets.
        max_cells_per_bucket:
            Maximum cells retained per bucket via reservoir sampling.
        store_cell_data:
            If ``True`` (default), accumulate sparse cell expression vectors.
            If ``False``, only cell IDs (obs_names) are stored; calling
            ``get_reservoir(return_cell_data=True)`` will raise.
        projector:
            Optional frozen encoder mapping ``(N, G) → (N, D)``. When given, its
            output feeds the LSH linear layer rather than raw gene expression.
            The module's gradients are disabled on assignment.
        seed:
            Random seed for the LSH projection weights.
    """

    def __init__(
        self,
        var_names_g: np.ndarray,
        n_bits: int = 12,
        max_cells_per_bucket: int = 100,
        store_cell_data: bool = True,
        projector: nn.Module | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()

        self.var_names_g = var_names_g
        self.n_bits = n_bits
        self.num_buckets = 2**n_bits
        self.max_cells_per_bucket = max_cells_per_bucket
        self.store_cell_data = store_cell_data
        self._seed = seed

        if projector is not None:
            self.projector: nn.Module | None = projector
            self.projector.requires_grad_(False)
        else:
            self.projector = None

        # DDP requires at least one parameter with requires_grad=True even when no
        # optimizer is used; this scalar satisfies that constraint without affecting results.
        self._dummy_param = nn.Parameter(torch.empty(()))

        # lsh_layer and data structures are created lazily on first forward() call.
        # _batches_seen is tracked here because global_step does not increment for
        # models that do not call optimizer.step() (automatic_optimization=False).
        self._batches_seen: int = 0
        self._prev_total_cells: int = 0
        self._ema_delta: float = 0.0

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _lazy_init(self, x_ng: torch.Tensor) -> None:
        """Build the LSH layer and bucket data structures on the first forward call."""
        if hasattr(self, "lsh_layer"):
            return

        device = x_ng.device

        if self.projector is not None:
            with torch.no_grad():
                sample = x_ng[:1].float()
                if sample.is_sparse:
                    sample = sample.to_dense()
                out = self.projector(sample)
            D = out.shape[1]
        else:
            D = len(self.var_names_g)

        self.lsh_layer = nn.Linear(D, self.n_bits, bias=False).to(device)
        with torch.no_grad():
            # Initialize on CPU for determinism across devices, then copy.
            gen = torch.Generator().manual_seed(self._seed)
            w = torch.empty(self.n_bits, D, dtype=torch.float32)
            nn.init.normal_(w, generator=gen)
            w /= w.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.lsh_layer.weight.data.copy_(w)
        self.lsh_layer.requires_grad_(False)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_bucket_ids(self, x_ng: torch.Tensor) -> torch.Tensor:
        """Project ``x_ng`` through the encoder (if any) and LSH layer → integer bucket IDs."""
        x = x_ng.float()
        if self.projector is not None:
            x = self.projector(x)
        logits = self.lsh_layer(x)  # (N, n_bits)
        bits = (logits > 0).long()
        powers = 2 ** torch.arange(self.n_bits, device=x.device, dtype=torch.long)
        return bits @ powers  # (N,)

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        obs_names_n: np.ndarray,
    ) -> dict[str, torch.Tensor | None]:
        """
        Accumulate a minibatch of cells into the geometric sketch reservoir.

        Args:
            x_ng:
                Expression matrix of shape ``(N, G)``, dense or sparse.
            var_names_g:
                Gene names for the batch; must match ``self.var_names_g``.
            obs_names_n:
                Cell IDs for the batch, shape ``(N,)``.

        Returns:
            An empty dict (no loss is computed).
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        self._lazy_init(x_ng)
        self.update(x_ng, obs_names_n)
        return {}

    @torch.no_grad()
    def update(self, x_ng: torch.Tensor, obs_names_n: np.ndarray) -> int:
        """
        Update the reservoir with a minibatch of cells.

        Args:
            x_ng:
                Expression matrix of shape ``(N, G)``, dense or sparse.
            obs_names_n:
                Cell IDs of shape ``(N,)``.

        Returns:
            Number of cells inserted or replaced in this update.
        """
        self._lazy_init(x_ng)
        x_float = x_ng.float()
        # Materialize a dense view for indexing; sparse indexing is not reliable across formats.
        x_dense = x_float.to_dense() if x_float.is_sparse else x_float

        bucket_ids = self._compute_bucket_ids(x_dense)
        inserted_count = 0

        for b_id in torch.unique(bucket_ids):
            b_idx = int(b_id.item())
            cell_indices = (bucket_ids == b_id).nonzero(as_tuple=True)[0]

            if b_idx not in self._bucket_total_seen:
                self._bucket_total_seen[b_idx] = 0
                self._bucket_obs_names[b_idx] = []
                if self.store_cell_data:
                    self._bucket_cells[b_idx] = []

            for idx in cell_indices:
                i = int(idx.item())
                seen = self._bucket_total_seen[b_idx]
                count = len(self._bucket_obs_names[b_idx])
                self._bucket_total_seen[b_idx] += 1
                obs_name = str(obs_names_n[i])

                if count < self.max_cells_per_bucket:
                    self._bucket_obs_names[b_idx].append(obs_name)
                    if self.store_cell_data:
                        self._bucket_cells[b_idx].append(x_dense[i].to_sparse())
                    inserted_count += 1
                else:
                    r = int(torch.randint(0, seen + 1, (1,)).item())
                    if r < self.max_cells_per_bucket:
                        self._bucket_obs_names[b_idx][r] = obs_name
                        if self.store_cell_data:
                            self._bucket_cells[b_idx][r] = x_dense[i].to_sparse()
                        inserted_count += 1

        return inserted_count

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_reservoir(self, return_cell_data: bool = True) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Retrieve the current sketch.

        Args:
            return_cell_data:
                If ``True`` (default), include sparse cell expression in the output.
                Requires ``store_cell_data=True`` at construction.

        Returns:
            A dict with:

            * ``"obs_names"`` — ``np.ndarray`` of cell IDs, always present.
            * ``"x_ng"`` — sparse CSR tensor of shape ``(N_sketch, G)``,
              present when ``return_cell_data=True``.
        """
        if return_cell_data and not self.store_cell_data:
            raise ValueError("store_cell_data=False was set at construction; cell expression data was not accumulated.")

        all_obs: list[str] = []
        all_cells: list[torch.Tensor] = []

        for b_idx in sorted(self._bucket_obs_names):
            all_obs.extend(self._bucket_obs_names[b_idx])
            if return_cell_data:
                all_cells.extend(self._bucket_cells[b_idx])

        result: dict[str, np.ndarray | torch.Tensor] = {"obs_names": np.array(all_obs)}

        if return_cell_data:
            if all_cells:
                result["x_ng"] = torch.stack([c.to_dense() for c in all_cells]).to_sparse_csr()
            else:
                result["x_ng"] = torch.zeros(0, len(self.var_names_g)).to_sparse_csr()

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cells(self) -> int:
        """Total cells currently stored in the sketch."""
        return sum(len(v) for v in self._bucket_obs_names.values())

    @property
    def num_filled_buckets(self) -> int:
        """Number of buckets that contain at least one cell."""
        return len(self._bucket_obs_names)

    @property
    def bucket_fill_fraction(self) -> float:
        """Fraction of the ``2^n_bits`` buckets that are occupied."""
        return self.num_filled_buckets / self.num_buckets

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            raise RuntimeError(
                f"{self.__class__.__name__} only supports single-device training "
                f"(got world_size={trainer.world_size}). Run on a single GPU or CPU."
            )

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        self._batches_seen += 1

        total = self.total_cells
        fill_frac = self.bucket_fill_fraction

        # EMA of per-batch growth in total_cells (not inserted_count, which also
        # counts reservoir replacements that don't change total_cells).
        delta = total - self._prev_total_cells
        self._ema_delta = 0.2 * delta + 0.8 * self._ema_delta
        self._prev_total_cells = total

        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("current_cells", float(total), prog_bar=True)
        trainer.model.log("fill_frac", fill_frac, prog_bar=True)

        n_total = trainer.num_training_batches
        if n_total != float("inf"):
            batches_remaining = n_total - self._batches_seen
            projected = total + self._ema_delta * batches_remaining
            trainer.model.log("proj_total_cells", projected, prog_bar=True)

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        trainer.should_stop = True

    # ------------------------------------------------------------------
    # Parameter reset
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """
        Reset all accumulated sketch data and re-seed the LSH projection weights.

        Safe to call before or after lazy initialization.
        """
        self._bucket_cells: dict[int, list[torch.Tensor]] = {}
        self._bucket_obs_names: dict[int, list[str]] = {}
        self._bucket_total_seen: dict[int, int] = {}
        self._batches_seen = 0
        self._prev_total_cells = 0
        self._ema_delta = 0.0

        if hasattr(self, "lsh_layer"):
            with torch.no_grad():
                D = self.lsh_layer.weight.shape[1]
                gen = torch.Generator().manual_seed(self._seed)
                w = torch.empty(self.n_bits, D, dtype=torch.float32)
                nn.init.normal_(w, generator=gen)
                w /= w.norm(dim=1, keepdim=True).clamp(min=1e-8)
                self.lsh_layer.weight.data.copy_(w)
            self.lsh_layer.requires_grad_(False)

        self._dummy_param.data.zero_()
