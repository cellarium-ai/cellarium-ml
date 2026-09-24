# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod

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
    Abstract base class for streaming geometric sketch models.

    Provides the shared reservoir data structures, :meth:`get_reservoir`,
    :meth:`apply_bucket_filters`, common Lightning hooks, and
    :meth:`reset_parameters`.  Concrete subclasses must implement
    :meth:`update` (the core hashing / binning algorithm) and
    :meth:`forward`.

    All bucket keys are ``tuple[int, ...]`` so that both 1-D (Hyperplane)
    and N-D (Plaid) coordinate schemes share the same type.

    Args:
        var_names_g:
            Gene names for input validation.
        max_cells_per_bucket:
            Maximum cells retained per bucket via uniform reservoir sampling.
        store_cell_data:
            If ``True``, accumulate sparse cell expression vectors.
            If ``False``, only cell IDs (``obs_names``) are stored; calling
            ``get_reservoir(return_cell_data=True)`` will raise.
        projector:
            Optional frozen encoder mapping ``(N, G) → (N, D)``. When given,
            its output feeds the bucketing algorithm rather than raw gene
            expression. The module's gradients are disabled on assignment.
        seed:
            Random seed for reservoir sampling. Sampling draws from a
            generator owned by this model, so results are unaffected by
            other consumers of the global :mod:`torch` RNG.
    """

    def __init__(
        self,
        var_names_g: np.ndarray,
        max_cells_per_bucket: int,
        store_cell_data: bool,
        projector: nn.Module | None,
        seed: int,
    ) -> None:
        super().__init__()

        self.var_names_g = var_names_g
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

        # Reservoir sampling draws from this generator rather than the global torch RNG,
        # so the sketch is reproducible regardless of other RNG consumers in the process.
        # It is seeded (and re-seeded) in reset_parameters().
        self._generator = torch.Generator()

        # _batches_seen is tracked here because global_step does not increment for
        # models that do not call optimizer.step() (automatic_optimization=False).
        # All bucket state dicts and EMA state are initialized by reset_parameters().
        self._prev_total_cells: int = 0
        self._ema_delta: float = 0.0
        self.reset_parameters()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def update(
        self,
        x_ng: torch.Tensor,
        obs_names_n: np.ndarray,
        metadata_n: np.ndarray | None = None,
    ) -> int:
        """
        Update the reservoir with a minibatch of cells.

        Args:
            x_ng:
                Expression matrix of shape ``(N, G)``, dense or sparse.
            obs_names_n:
                Cell IDs of shape ``(N,)``.
            metadata_n:
                Optional integer metadata of shape ``(N,)`` used for
                diversity tracking in :meth:`apply_bucket_filters`.

        Returns:
            Number of cells inserted or replaced in this update.
        """

    # ------------------------------------------------------------------
    # Retrieval and filtering
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_reservoir(
        self,
        return_cell_data: bool = False,
        max_cells: int | None = None,
        seed: int | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Retrieve the current sketch.

        Buckets are visited in sorted key order so the pre-downsample
        ordering is deterministic.  This method is non-destructive; call
        :meth:`apply_bucket_filters` separately to prune the internal state.

        Args:
            return_cell_data:
                If ``True``, include sparse cell expression in the output.
                Requires ``store_cell_data=True`` at construction.
            max_cells:
                If given, randomly downsample the result to at most this
                many cells.  When the reservoir is already smaller than
                ``max_cells`` no downsampling occurs.
            seed:
                Random seed for the ``max_cells`` downsample.  ``None``
                gives a non-reproducible draw.

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

        for b_key in sorted(self._bucket_obs_names):
            all_obs.extend(self._bucket_obs_names[b_key])
            if return_cell_data:
                all_cells.extend(self._bucket_cells[b_key])

        if max_cells is not None and len(all_obs) > max_cells:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(all_obs), size=max_cells, replace=False))
            all_obs = [all_obs[i] for i in indices]
            if return_cell_data:
                all_cells = [all_cells[i] for i in indices]

        result: dict[str, np.ndarray | torch.Tensor] = {"obs_names": np.array(all_obs)}

        if return_cell_data:
            if all_cells:
                result["x_ng"] = torch.stack([c.to_dense() for c in all_cells]).to_sparse_csr()
            else:
                result["x_ng"] = torch.zeros(0, len(self.var_names_g)).to_sparse_csr()

        return result

    def apply_bucket_filters(
        self,
        min_cells_per_bucket: int = 1,
        min_metadata_diversity: int = 1,
    ) -> None:
        """
        Prune buckets in-place that fail density or diversity thresholds.

        Uses ``_bucket_total_seen`` (the total number of cells ever hashed
        to a bucket, not the capped retained count) as the density measure.
        This correctly identifies sparse regions even when ``max_cells_per_bucket``
        caps the retained count below the threshold.

        Args:
            min_cells_per_bucket:
                Buckets where fewer than this many cells were ever observed
                are deleted.
            min_metadata_diversity:
                Buckets that observed fewer than this many unique metadata
                categories (from ``metadata_n``) are deleted.  Ignored
                when ``min_metadata_diversity <= 1``.

        Raises:
            ValueError:
                If ``min_metadata_diversity > 1`` but no metadata was ever
                tracked (i.e. ``metadata_n`` was never passed to
                :meth:`update` or :meth:`forward`).
        """
        if (
            min_metadata_diversity > 1
            and self._bucket_metadata
            and all(len(v) == 0 for v in self._bucket_metadata.values())
        ):
            raise ValueError(
                "min_metadata_diversity > 1 but no metadata was tracked. Pass metadata_n to forward() or update()."
            )

        keys_to_delete = []
        for k, seen in self._bucket_total_seen.items():
            if seen < min_cells_per_bucket:
                keys_to_delete.append(k)
                continue
            if min_metadata_diversity > 1 and len(self._bucket_metadata[k]) < min_metadata_diversity:
                keys_to_delete.append(k)

        for k in keys_to_delete:
            del self._bucket_total_seen[k]
            del self._bucket_obs_names[k]
            del self._bucket_metadata[k]
            if self.store_cell_data:
                del self._bucket_cells[k]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cells(self) -> int:
        """Total cells currently stored in the sketch."""
        return sum(len(v) for v in self._bucket_obs_names.values())

    @property
    def num_filled_buckets(self) -> int:
        """Number of buckets containing at least one cell."""
        return len(self._bucket_obs_names)

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
        delta = total - self._prev_total_cells
        self._ema_delta = 0.2 * delta + 0.8 * self._ema_delta
        self._prev_total_cells = total

        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("current_cells", float(total), prog_bar=True)

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
        self._bucket_cells: dict[tuple[int, ...], list[torch.Tensor]] = {}
        self._bucket_obs_names: dict[tuple[int, ...], list[str]] = {}
        self._bucket_total_seen: dict[tuple[int, ...], int] = {}
        self._bucket_metadata: dict[tuple[int, ...], set[int]] = {}
        self._batches_seen = 0
        self._prev_total_cells = 0
        self._ema_delta = 0.0
        self._generator.manual_seed(self._seed)
        self._dummy_param.data.zero_()


class StreamingHyperplaneGeometricSketch(StreamingGeometricSketch):
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
            Random seed for the LSH projection weights and for reservoir sampling.
            Sampling draws from a generator owned by this model, so results are
            unaffected by other consumers of the global :mod:`torch` RNG.
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
        # Set before super().__init__() so reset_parameters() can reference them.
        self.n_bits = n_bits
        self.num_buckets = 2**n_bits
        super().__init__(
            var_names_g=var_names_g,
            max_cells_per_bucket=max_cells_per_bucket,
            store_cell_data=store_cell_data,
            projector=projector,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _lazy_init(self, x_ng: torch.Tensor) -> None:
        """Build the LSH layer on the first forward call."""
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
        metadata_n: np.ndarray | None = None,
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
            metadata_n:
                Optional integer metadata of shape ``(N,)`` for diversity
                tracking.  When provided, each cell's value is recorded in
                its bucket's metadata set, enabling :meth:`apply_bucket_filters`
                with ``min_metadata_diversity > 1``.

        Returns:
            An empty dict (no loss is computed).
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        self._lazy_init(x_ng)
        self.update(x_ng, obs_names_n, metadata_n)
        return {}

    @torch.no_grad()
    def update(
        self,
        x_ng: torch.Tensor,
        obs_names_n: np.ndarray,
        metadata_n: np.ndarray | None = None,
    ) -> int:
        """
        Update the reservoir with a minibatch of cells.

        Args:
            x_ng:
                Expression matrix of shape ``(N, G)``, dense or sparse.
            obs_names_n:
                Cell IDs of shape ``(N,)``.
            metadata_n:
                Optional integer metadata of shape ``(N,)``.

        Returns:
            Number of cells inserted or replaced in this update.
        """
        x_float = x_ng.float()
        x_dense = x_float.to_dense() if x_float.is_sparse else x_float

        bucket_ids = self._compute_bucket_ids(x_dense)
        inserted_count = 0

        for b_id in torch.unique(bucket_ids):
            b_key = (int(b_id.item()),)
            cell_indices = (bucket_ids == b_id).nonzero(as_tuple=True)[0]

            if b_key not in self._bucket_total_seen:
                self._bucket_total_seen[b_key] = 0
                self._bucket_obs_names[b_key] = []
                self._bucket_metadata[b_key] = set()
                if self.store_cell_data:
                    self._bucket_cells[b_key] = []

            for idx in cell_indices:
                i = int(idx.item())
                seen = self._bucket_total_seen[b_key]
                count = len(self._bucket_obs_names[b_key])
                self._bucket_total_seen[b_key] += 1
                obs_name = str(obs_names_n[i])

                if metadata_n is not None:
                    self._bucket_metadata[b_key].add(int(metadata_n[i]))

                if count < self.max_cells_per_bucket:
                    self._bucket_obs_names[b_key].append(obs_name)
                    if self.store_cell_data:
                        self._bucket_cells[b_key].append(x_dense[i].to_sparse())
                    inserted_count += 1
                else:
                    r = int(torch.randint(0, seen + 1, (1,), generator=self._generator).item())
                    if r < self.max_cells_per_bucket:
                        self._bucket_obs_names[b_key][r] = obs_name
                        if self.store_cell_data:
                            self._bucket_cells[b_key][r] = x_dense[i].to_sparse()
                        inserted_count += 1

        return inserted_count

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bucket_fill_fraction(self) -> float:
        """Fraction of the ``2^n_bits`` buckets that are occupied."""
        return self.num_filled_buckets / self.num_buckets

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        super().on_train_batch_end(trainer)
        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("fill_frac", self.bucket_fill_fraction, prog_bar=True)

    # ------------------------------------------------------------------
    # Parameter reset
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """
        Reset all accumulated sketch data and re-seed the LSH projection weights.

        Safe to call before or after lazy initialization.
        """
        super().reset_parameters()
        if hasattr(self, "lsh_layer"):
            with torch.no_grad():
                D = self.lsh_layer.weight.shape[1]
                gen = torch.Generator().manual_seed(self._seed)
                w = torch.empty(self.n_bits, D, dtype=torch.float32)
                nn.init.normal_(w, generator=gen)
                w /= w.norm(dim=1, keepdim=True).clamp(min=1e-8)
                self.lsh_layer.weight.data.copy_(w)
            self.lsh_layer.requires_grad_(False)


class StreamingPlaidGeometricSketch(StreamingGeometricSketch):
    r"""
    Online geometric sketching [1] via Dynamic Spatial Hashing.

    Streams single-cell gene expression data and retains a geometrically diverse
    sketch of cells across a single pass. It constructs a true plaid $\epsilon$-cover
    of the latent space. As the stream progresses, the spatial grid dynamically coarsens
    (doubling the voxel size) whenever the number of occupied voxels exceeds the target,
    re-merging local reservoirs perfectly.

    To combat technical artifacts, an end-of-epoch pruning step removes voxels
    that fail to meet minimum cell count (density) or minimum categorical diversity
    (e.g., number of unique datasets or patients) thresholds.

    Only single-device training is supported. A ``RuntimeError`` is raised at the
    start of training if more than one device is detected.

    References:
        [1] Hie, B., ..., Berger, B. (2019). Geometric sketching compactly summarizes
            the single-cell transcriptomic landscape. Cell Systems, 8(6), 483-493.e7.

    Args:
        var_names_g:
            Gene names for input validation.
        target_voxels:
            The threshold for spatial grid coarsening.
        initial_voxel_size:
            Starting size ($\epsilon$) of the grid hypercubes.
        max_cells_per_bucket:
            Maximum cells retained per voxel via uniform reservoir sampling.
        min_cells_per_voxel:
            Density threshold applied at the end of training via
            :meth:`apply_bucket_filters`. Voxels that observed fewer cells
            in total are dropped.
        min_metadata_diversity:
            Diversity threshold applied at the end of training via
            :meth:`apply_bucket_filters`. Voxels that observed fewer unique
            metadata categories (from ``metadata_n``) are dropped.
        store_cell_data:
            If ``True``, accumulate sparse cell expression vectors.
        projector:
            Optional frozen encoder mapping ``(N, G) → (N, D)``.
        seed:
            Random seed for reservoir sampling and reservoir merging. Sampling
            draws from a generator owned by this model, so results are unaffected
            by other consumers of the global :mod:`torch` RNG.
    """

    def __init__(
        self,
        var_names_g: np.ndarray,
        target_voxels: int = 100_000,
        initial_voxel_size: float = 0.1,
        max_cells_per_bucket: int = 1,
        min_cells_per_voxel: int = 5,
        min_metadata_diversity: int = 1,
        store_cell_data: bool = False,
        projector: nn.Module | None = None,
        seed: int = 0,
    ) -> None:
        # Set before super().__init__() so reset_parameters() and on_train_epoch_end() can reference them.
        self.target_voxels = target_voxels
        self.initial_voxel_size = initial_voxel_size
        self.min_cells_per_voxel = min_cells_per_voxel
        self.min_metadata_diversity = min_metadata_diversity

        super().__init__(
            var_names_g=var_names_g,
            max_cells_per_bucket=max_cells_per_bucket,
            store_cell_data=store_cell_data,
            projector=projector,
            seed=seed,
        )

        # Registered after super().__init__() (which calls reset_parameters()); the
        # reset_parameters() override checks hasattr so it safely skips on first call.
        self.register_buffer("voxel_size", torch.tensor(initial_voxel_size, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _merge_reservoirs(
        self,
        obs_A: list[str],
        cells_A: list[torch.Tensor],
        seen_A: int,
        obs_B: list[str],
        cells_B: list[torch.Tensor],
        seen_B: int,
    ) -> tuple[list[str], list[torch.Tensor], int]:
        total_seen = seen_A + seen_B
        K = self.max_cells_per_bucket

        if total_seen <= K:
            return obs_A + obs_B, cells_A + cells_B, total_seen

        prob_A = seen_A / total_seen
        pick_A = int(
            torch.binomial(
                torch.tensor(float(K)), torch.tensor(prob_A, dtype=torch.float32), generator=self._generator
            ).item()
        )
        pick_B = K - pick_A

        pick_A = min(pick_A, len(obs_A))
        pick_B = min(pick_B, len(obs_B))

        while pick_A + pick_B < K:
            if pick_A < len(obs_A):
                pick_A += 1
            elif pick_B < len(obs_B):
                pick_B += 1
            else:
                break

        idx_A = torch.randperm(len(obs_A), generator=self._generator)[:pick_A]
        idx_B = torch.randperm(len(obs_B), generator=self._generator)[:pick_B]

        new_obs = [obs_A[i] for i in idx_A] + [obs_B[i] for i in idx_B]
        new_cells = []
        if self.store_cell_data:
            new_cells = [cells_A[i] for i in idx_A] + [cells_B[i] for i in idx_B]

        return new_obs, new_cells, total_seen

    @torch.no_grad()
    def _coarsen(self) -> None:
        self.voxel_size *= 2.0

        new_total_seen: dict[tuple[int, ...], int] = {}
        new_obs_names: dict[tuple[int, ...], list[str]] = {}
        new_metadata: dict[tuple[int, ...], set[int]] = {}
        new_cells: dict[tuple[int, ...], list[torch.Tensor]] = {}

        for old_coord, seen in self._bucket_total_seen.items():
            new_coord = tuple(c // 2 for c in old_coord)
            obs = self._bucket_obs_names[old_coord]
            meta = self._bucket_metadata[old_coord]
            cells = self._bucket_cells.get(old_coord, [])

            if new_coord not in new_total_seen:
                new_total_seen[new_coord] = seen
                new_obs_names[new_coord] = obs
                new_metadata[new_coord] = set(meta)
                if self.store_cell_data:
                    new_cells[new_coord] = cells
            else:
                m_obs, m_cells, m_seen = self._merge_reservoirs(
                    new_obs_names[new_coord], new_cells.get(new_coord, []), new_total_seen[new_coord], obs, cells, seen
                )
                new_total_seen[new_coord] = m_seen
                new_obs_names[new_coord] = m_obs
                new_metadata[new_coord].update(meta)
                if self.store_cell_data:
                    new_cells[new_coord] = m_cells

        self._bucket_total_seen = new_total_seen
        self._bucket_obs_names = new_obs_names
        self._bucket_metadata = new_metadata
        if self.store_cell_data:
            self._bucket_cells = new_cells

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        obs_names_n: np.ndarray,
        metadata_n: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor | None]:
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        if self.min_metadata_diversity > 1 and metadata_n is None:
            raise ValueError("metadata_n must be provided when min_metadata_diversity > 1.")

        self.update(x_ng, obs_names_n, metadata_n)
        return {}

    @torch.no_grad()
    def update(
        self,
        x_ng: torch.Tensor,
        obs_names_n: np.ndarray,
        metadata_n: np.ndarray | None = None,
    ) -> int:
        x_float = x_ng.float()
        x_dense = x_float.to_dense() if x_float.is_sparse else x_float

        if self.projector is not None:
            z = self.projector(x_dense)
        else:
            z = x_dense

        coords = torch.floor(z / self.voxel_size).long()
        unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
        inserted_count = 0

        unique_coords_np = unique_coords.cpu().numpy()
        inverse_indices_np = inverse_indices.cpu().numpy()

        for ui, coord in enumerate(unique_coords_np):
            coord_tuple = tuple(coord)
            cell_indices = (inverse_indices_np == ui).nonzero()[0]

            if coord_tuple not in self._bucket_total_seen:
                self._bucket_total_seen[coord_tuple] = 0
                self._bucket_obs_names[coord_tuple] = []
                self._bucket_metadata[coord_tuple] = set()
                if self.store_cell_data:
                    self._bucket_cells[coord_tuple] = []

            for i in cell_indices:
                seen = self._bucket_total_seen[coord_tuple]
                count = len(self._bucket_obs_names[coord_tuple])
                self._bucket_total_seen[coord_tuple] += 1
                obs_name = str(obs_names_n[i])

                if metadata_n is not None:
                    self._bucket_metadata[coord_tuple].add(int(metadata_n[i]))

                if count < self.max_cells_per_bucket:
                    self._bucket_obs_names[coord_tuple].append(obs_name)
                    if self.store_cell_data:
                        self._bucket_cells[coord_tuple].append(x_dense[i].to_sparse())
                    inserted_count += 1
                else:
                    r = int(torch.randint(0, seen + 1, (1,), generator=self._generator).item())
                    if r < self.max_cells_per_bucket:
                        self._bucket_obs_names[coord_tuple][r] = obs_name
                        if self.store_cell_data:
                            self._bucket_cells[coord_tuple][r] = x_dense[i].to_sparse()
                        inserted_count += 1

        if len(self._bucket_total_seen) > self.target_voxels:
            self._coarsen()

        return inserted_count

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        super().on_train_batch_end(trainer)
        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("voxel_size", self.voxel_size.item(), prog_bar=True)
        trainer.model.log("active_voxels", float(self.num_filled_buckets), prog_bar=True)

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        super().on_train_epoch_end(trainer)
        self.apply_bucket_filters(self.min_cells_per_voxel, self.min_metadata_diversity)
        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("final_sketch_size", float(self.total_cells))

    # ------------------------------------------------------------------
    # Parameter reset
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "voxel_size"):
            self.voxel_size.fill_(self.initial_voxel_size)
