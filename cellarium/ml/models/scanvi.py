# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""scANVI: Single-cell Annotation using Variational Inference."""

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as kl

from cellarium.ml.layers import DressedLayer, FullyConnectedLinear
from cellarium.ml.models.scvi import (
    SingleCellVariationalInference,
    compute_annealed_kl_weight,
    weights_init,
)
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class ConditionalNormalEncoder(torch.nn.Module):
    """Encode concatenated [z, one_hot(c)] to a Normal distribution.

    Expects 2D input ``[N, in_features]``. The caller is responsible for reshaping
    3D inputs ``[N, chunk, in_features]`` to ``[N*chunk, in_features]`` before calling
    and reshaping the output back.

    Args:
        in_features: Input dimensionality (typically ``n_latent + n_classes``).
        out_features: Output (latent) dimensionality.
        n_hidden: Widths of hidden layers. Empty list gives a single linear layer.
        use_batch_norm: Whether to apply ``BatchNorm1d`` in hidden layers.
        dropout_rate: Dropout rate in hidden layers.
        var_eps: Floor added to the variance for numerical stability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_hidden: list[int],
        use_batch_norm: bool = True,
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
                    use_batch_norm=use_batch_norm,
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


class SCANVI(SingleCellVariationalInference):
    """Single-cell ANnotation using Variational Inference (scANVI) [1].

    Extends scVI with a hierarchical prior on the latent variable ``z``:

    .. math::

        c \\sim p(c), \\quad u \\sim \\mathcal{N}(0, I),
        \\quad z \\sim p(z \\mid u, c), \\quad x \\sim p(x \\mid z, s)

    The encoder ``q(z|x)`` is identical to scVI. Three new networks are added:

    * **Classifier** ``q(c|z)``: predicts cell-type probabilities from ``z``.
    * **u-encoder** ``q(u|z,c)``: encodes a deeper latent ``u`` given ``z`` and cell type ``c``.
    * **z-prior decoder** ``p(z|u,c)``: provides the class-conditional prior on ``z``.

    For labeled cells the ELBO is maximized using the known cell type together with an
    auxiliary cross-entropy classification term.  For unlabeled cells the cell-type
    expectation is marginalized in chunks to control peak VRAM.

    .. note::

        At large ``n_classes`` (> ~500) the one-hot encoding of ``c`` dominates the input
        to the secondary networks.  A learned cell-type embedding is a planned extension.

    **References:**

    1. `Probabilistic harmonization and annotation of single-cell transcriptomics data
       with deep generative models (Xu et al., 2021)
       <https://doi.org/10.15252/msb.20209620>`_.

    Args:
        n_classes:
            Number of annotated cell-type classes.
        unlabeled_category_index:
            Integer code used to mark unlabeled cells in ``cell_type_index_n``. Default ``-1``.
        classification_weight:
            Weight ``α`` applied to the cross-entropy loss on labeled cells.
        chunk_size:
            Number of classes processed per chunk during the unlabeled forward pass.
            Reduce this to lower peak VRAM at the cost of slightly more computation.
        classifier_n_hidden:
            Hidden layer widths for the classifier ``q(c|z)`` MLP.
            Defaults to ``[128]`` (one hidden layer, matching scvi-tools).
        classifier_dropout_rate:
            Dropout rate in classifier hidden layers.
        secondary_n_hidden:
            Hidden layer widths shared by the u-encoder ``q(u|z,c)`` and the
            z-prior decoder ``p(z|u,c)``. Defaults to ``[128]``.
        y_prior_probs:
            Prior probabilities over cell types ``p(c)``.  Must sum to 1 and have length
            ``n_classes``. If ``None`` a uniform prior is used.
        **scvi_kwargs:
            All keyword arguments accepted by
            :class:`~cellarium.ml.models.SingleCellVariationalInference`.
    """

    def __init__(
        self,
        n_classes: int,
        unlabeled_category_index: int = -1,
        classification_weight: float = 50.0,
        chunk_size: int = 100,
        classifier_n_hidden: list[int] | None = None,
        classifier_dropout_rate: float = 0.1,
        secondary_n_hidden: list[int] | None = None,
        y_prior_probs: list[float] | None = None,
        **scvi_kwargs,
    ):
        if scvi_kwargs.get("use_flow", False):
            raise ValueError("SCANVI does not support use_flow=True.")

        super().__init__(**scvi_kwargs)

        if classifier_n_hidden is None:
            classifier_n_hidden = [128]
        if secondary_n_hidden is None:
            secondary_n_hidden = [128]

        self.n_classes = n_classes
        self.unlabeled_category_index = unlabeled_category_index
        self.classification_weight = classification_weight
        self.chunk_size = chunk_size

        # Verify n_classes is consistent with cell_type_categories when provided
        cell_type_categories = scvi_kwargs.get("cell_type_categories")
        if cell_type_categories is not None and len(cell_type_categories) != n_classes:
            raise ValueError(
                f"n_classes={n_classes} does not match len(cell_type_categories)={len(cell_type_categories)}."
            )

        # q(c|z): MLP classifier from latent space to cell-type logits
        self.cell_type_classifier = FullyConnectedLinear(
            in_features=self.n_latent,
            out_features=n_classes,
            n_hidden=classifier_n_hidden,
            dressing_init_kwargs={"use_batch_norm": True, "dropout_rate": classifier_dropout_rate},
            bias=True,
        )

        # q(u|z,c): encodes the deeper latent u given z and cell type
        self.u_encoder = ConditionalNormalEncoder(
            in_features=self.n_latent + n_classes,
            out_features=self.n_latent,
            n_hidden=secondary_n_hidden,
        )

        # p(z|u,c): class-conditional prior on z, parameterized by u and cell type
        self.z_prior_decoder = ConditionalNormalEncoder(
            in_features=self.n_latent + n_classes,
            out_features=self.n_latent,
            n_hidden=secondary_n_hidden,
        )

        # Prior over cell types p(c); uniform by default
        if y_prior_probs is not None:
            y_prior = torch.tensor(y_prior_probs, dtype=torch.float32)
            y_prior = y_prior / y_prior.sum()
        else:
            y_prior = torch.ones(n_classes, dtype=torch.float32) / n_classes
        self.register_buffer("y_prior", y_prior)

        # Initialize new modules (parent's reset_parameters() ran before these existed)
        self.cell_type_classifier.apply(weights_init)
        self.u_encoder.apply(weights_init)
        self.z_prior_decoder.apply(weights_init)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _compute_conditional_kls(
        self,
        z: torch.Tensor,
        qz_mean: torch.Tensor,
        qz_std: torch.Tensor,
        c_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``KL(q(z|x) || p(z|u,c))`` and ``KL(q(u|z,c) || N(0,I))`` for cell-class pairs.

        All inputs must be 2D: ``[N, ...]``.  The caller reshapes from 3D when batching over
        the class dimension.

        Args:
            z: Latent sample from ``q(z|x)``, shape ``[N, n_latent]``.
            qz_mean: Mean of ``q(z|x)``, shape ``[N, n_latent]``.
            qz_std: Std of ``q(z|x)``, shape ``[N, n_latent]``.
            c_onehot: One-hot cell-type encoding, shape ``[N, n_classes]``.

        Returns:
            ``kl_z``: ``[N]`` tensor — KL divergence for z.
            ``kl_u``: ``[N]`` tensor — KL divergence for u.
        """
        zc = torch.cat([z, c_onehot], dim=-1)
        qu = self.u_encoder(zc)
        u = qu.rsample()

        uc = torch.cat([u, c_onehot], dim=-1)
        pz = self.z_prior_decoder(uc)

        qz = Normal(qz_mean, qz_std)
        pu = Normal(torch.zeros_like(u), torch.ones_like(u))

        kl_z = kl(qz, pz).sum(dim=-1)
        kl_u = kl(qu, pu).sum(dim=-1)
        return kl_z, kl_u

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
        cell_type_index_n: torch.Tensor | None = None,
    ) -> dict:
        """Compute scANVI ELBO and auxiliary classification loss.

        Args:
            x_ng: Gene counts matrix ``[N, G]``.
            var_names_g: Variable names for input validation.
            batch_index_n: Integer batch indices ``[N]``.
            continuous_covariates_nc: Continuous covariates ``[N, C]``.
            categorical_covariate_index_nd: Integer categorical covariate codes ``[N, D]``.
            total_mrna_umis_n: Total mRNA UMIs per cell (not log-scaled).
            cell_type_index_n: Integer cell-type codes ``[N]``.  Cells with code equal to
                ``unlabeled_category_index`` (default ``-1``) are treated as unlabeled.
                If ``None``, all cells are treated as unlabeled.

        Returns:
            A dictionary with keys:
                - ``"loss"``: Scalar training loss.
                - ``"reconstruction_loss"``: Per-cell NLL ``[N]``.
                - ``"kl_divergence_z"``: Per-cell ``KL(q(z|x) || p(z|u,c))`` ``[N]``.
                - ``"kl_divergence_u"``: Per-cell ``KL(q(u|z,c) || N(0,I))`` ``[N]``.
                - ``"kl_divergence_c"``: Per-cell ``KL(q(c|z) || p(c))`` (unlabeled only) ``[N]``.
                - ``"kl_divergence_batch"``: Per-cell batch-embedding KL (if configured) ``[N]``.
                - ``"classification_loss"``: Per-cell cross-entropy (labeled only) ``[N]``.
                - ``"z_nk"``: Latent samples ``[N, n_latent]``.
                - ``"cell_type_logits_nc"``: Raw classifier logits ``[N, n_classes]``.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        n = x_ng.shape[0]
        device = x_ng.device

        if cell_type_index_n is None:
            cell_type_index_n = torch.full((n,), self.unlabeled_category_index, dtype=torch.long, device=device)

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

        # --- scVI encoder q(z|x) and decoder p(x|z) — unchanged from parent ---
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

        z = inference_outputs["z"]  # [N, n_latent]
        qz = inference_outputs["qz"]  # Normal with batch_shape [N, n_latent]
        qz_mean = qz.mean  # [N, n_latent]
        qz_std = qz.stddev  # [N, n_latent]

        rec_loss_n = -generative_outputs["px"].log_prob(x_ng).sum(dim=-1)  # [N]

        # KL annealing weight — applied to kl_z and kl_u only, per user spec
        kl_annealing_weight = compute_annealed_kl_weight(
            epoch=self.epoch,
            step=self.step,
            n_epochs_kl_warmup=self.kl_warmup_epochs,
            n_steps_kl_warmup=self.kl_warmup_steps,
            max_kl_weight=1.0,
            min_kl_weight=self.kl_annealing_start,
        )

        # Optional batch-embedding KL (inherited from scVI; annealed with kl_z/kl_u)
        kl_divergence_batch_n = torch.zeros(n, device=device)
        if self.batch_representation_sampled and (self.batch_kl_weight_max > 0):
            kl_divergence_batch_n = kl(
                self.batch_embedding_distribution(batch_index_n=batch_index_n),
                Normal(torch.zeros_like(batch_nb), torch.ones_like(batch_nb)),
            ).sum(dim=1)

        # --- Classifier q(c|z): use qz.mean for stability ---
        logits_nc = self.cell_type_classifier(qz_mean)  # [N, n_classes]
        probs_nc = F.softmax(logits_nc, dim=-1)  # [N, n_classes]

        # --- Labeled / unlabeled split ---
        labeled_mask = cell_type_index_n != self.unlabeled_category_index
        unlabeled_mask = ~labeled_mask

        kl_z_n = torch.zeros(n, device=device)
        kl_u_n = torch.zeros(n, device=device)
        kl_c_n = torch.zeros(n, device=device)
        ce_loss_n = torch.zeros(n, device=device)

        # --- Labeled cells: use known class, add cross-entropy loss ---
        if labeled_mask.any():
            idx_lab = labeled_mask.nonzero(as_tuple=True)[0]
            true_labels = cell_type_index_n[idx_lab]  # [N_lab]
            c_true = F.one_hot(true_labels, self.n_classes).float()  # [N_lab, n_classes]

            kl_z_lab, kl_u_lab = self._compute_conditional_kls(
                z=z[idx_lab],
                qz_mean=qz_mean[idx_lab],
                qz_std=qz_std[idx_lab],
                c_onehot=c_true,
            )
            kl_z_n[idx_lab] = kl_z_lab
            kl_u_n[idx_lab] = kl_u_lab
            ce_loss_n[idx_lab] = F.cross_entropy(logits_nc[idx_lab], true_labels, reduction="none")

        # --- Unlabeled cells: marginalize over classes in chunks ---
        if unlabeled_mask.any():
            idx_unl = unlabeled_mask.nonzero(as_tuple=True)[0]
            n_unl = idx_unl.shape[0]

            z_unl = z[idx_unl]  # [N_unl, n_latent]
            qz_mean_unl = qz_mean[idx_unl]  # [N_unl, n_latent]
            qz_std_unl = qz_std[idx_unl]  # [N_unl, n_latent]
            probs_unl = probs_nc[idx_unl]  # [N_unl, n_classes]

            kl_z_sum = torch.zeros(n_unl, device=device)
            kl_u_sum = torch.zeros(n_unl, device=device)

            for start in range(0, self.n_classes, self.chunk_size):
                end = min(start + self.chunk_size, self.n_classes)
                chunk = end - start

                # One-hot vectors for this chunk's classes: [chunk, n_classes]
                c_chunk = F.one_hot(torch.arange(start, end, device=device), self.n_classes).float()

                # Expand to [N_unl, chunk, ...]
                c_expanded = c_chunk.unsqueeze(0).expand(n_unl, -1, -1)  # [N_unl, chunk, n_classes]
                z_expanded = z_unl.unsqueeze(1).expand(-1, chunk, -1)  # [N_unl, chunk, n_latent]
                qz_mean_exp = qz_mean_unl.unsqueeze(1).expand(-1, chunk, -1)  # [N_unl, chunk, n_latent]
                qz_std_exp = qz_std_unl.unsqueeze(1).expand(-1, chunk, -1)  # [N_unl, chunk, n_latent]

                # Reshape to 2D for network forward passes (avoids BatchNorm1d shape issues)
                nc = n_unl * chunk
                zc_2d = torch.cat([z_expanded, c_expanded], dim=-1).reshape(nc, -1)
                c_2d = c_expanded.reshape(nc, -1)

                qu_chunk = self.u_encoder(zc_2d)  # Normal([nc, n_latent])
                u_sample = qu_chunk.rsample()  # [nc, n_latent]

                pz_chunk = self.z_prior_decoder(torch.cat([u_sample, c_2d], dim=-1))  # Normal([nc, n_latent])

                qz_chunk = Normal(qz_mean_exp.reshape(nc, -1), qz_std_exp.reshape(nc, -1))
                pu_chunk = Normal(torch.zeros_like(u_sample), torch.ones_like(u_sample))

                # Per-chunk KLs, reshaped to [N_unl, chunk]
                kl_z_chunk = kl(qz_chunk, pz_chunk).sum(dim=-1).view(n_unl, chunk)
                kl_u_chunk = kl(qu_chunk, pu_chunk).sum(dim=-1).view(n_unl, chunk)

                # Weight by predicted class probabilities and accumulate
                probs_chunk = probs_unl[:, start:end]  # [N_unl, chunk]
                kl_z_sum += (probs_chunk * kl_z_chunk).sum(dim=-1)
                kl_u_sum += (probs_chunk * kl_u_chunk).sum(dim=-1)

            kl_z_n[idx_unl] = kl_z_sum
            kl_u_n[idx_unl] = kl_u_sum

            # KL(q(c|z) || p(c)) for unlabeled cells — not annealed
            kl_c_n[idx_unl] = kl(
                Categorical(probs=probs_unl),
                Categorical(probs=self.y_prior.unsqueeze(0).expand(n_unl, -1)),
            )

        # --- Loss assembly ---
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
            "cell_type_logits_nc": logits_nc,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
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
        validation_cell_type_index_n: torch.Tensor | None = None,
        cell_type_index_n: torch.Tensor | None = None,
    ) -> None:
        n = x_ng.shape[0]

        output = self(
            x_ng=x_ng,
            var_names_g=var_names_g,
            batch_index_n=batch_index_n,
            cell_type_index_n=cell_type_index_n,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_index_nd=categorical_covariate_index_nd,
            total_mrna_umis_n=total_mrna_umis_n,
        )

        if isinstance(output["loss"], torch.Tensor):
            pl_module.log("val_loss", output["loss"], sync_dist=True, on_epoch=True, batch_size=n)

        # Full ELBO includes kl_u and kl_c (unlike base scVI)
        kl_batch = output["kl_divergence_batch"]
        rec_loss = output["reconstruction_loss"]
        kl_z_val = output["kl_divergence_z"]
        kl_u_val = output["kl_divergence_u"]
        kl_c_val = output["kl_divergence_c"]
        z_nk = output["z_nk"]
        assert isinstance(kl_batch, torch.Tensor)
        assert isinstance(rec_loss, torch.Tensor)
        assert isinstance(kl_z_val, torch.Tensor)
        assert isinstance(kl_u_val, torch.Tensor)
        assert isinstance(kl_c_val, torch.Tensor)
        assert isinstance(z_nk, torch.Tensor)

        elbo_n = -(rec_loss + kl_z_val + kl_u_val + kl_c_val + kl_batch)
        self._val_elbo_sum += elbo_n.sum().detach()
        self._val_rec_sum += rec_loss.sum().detach()
        self._val_n_cells += n

        z_nk = z_nk.detach()

        # Per-class z sums for ontology metric (uses validation labels, not training labels)
        if validation_cell_type_index_n is not None and self.num_classes is not None:
            idx = validation_cell_type_index_n.long()
            self._val_z_sum_kd.index_add_(0, idx, z_nk)
            self._val_class_count_k.index_add_(0, idx, torch.ones(n, device=x_ng.device))

        # Per-batch z sums for batch silhouette metric
        if self.n_batch > 1:
            bidx = batch_index_n.long()
            self._val_batch_z_sum_bk.index_add_(0, bidx, z_nk)
            self._val_batch_z_sq_sum_b.index_add_(0, bidx, (z_nk**2).sum(dim=-1))
            self._val_batch_count_b.index_add_(0, bidx, torch.ones(n, device=x_ng.device))

        # Reservoir sampling for cell-type logistic-regression classifier
        if validation_cell_type_index_n is not None and self.num_classes is not None:
            if batch_idx % 2 == 0:
                buf_z, buf_y = self._val_cl_train_z, self._val_cl_train_y
                fill, seen = self._val_cl_train_fill, self._val_cl_train_seen
            else:
                buf_z, buf_y = self._val_cl_test_z, self._val_cl_test_y
                fill, seen = self._val_cl_test_fill, self._val_cl_test_seen

            rs = self.val_cell_type_classifier_reservoir_size
            labels = validation_cell_type_index_n.long()

            space = rs - fill
            direct = min(space, n)
            if direct > 0:
                buf_z[fill : fill + direct] = z_nk[:direct]
                buf_y[fill : fill + direct] = labels[:direct]
                fill += direct

            for i in range(direct, n):
                j = int(torch.randint(0, seen + i - direct + 1, (1,)).item())
                if j < rs:
                    buf_z[j] = z_nk[i]
                    buf_y[j] = labels[i]

            seen += n
            if batch_idx % 2 == 0:
                self._val_cl_train_fill, self._val_cl_train_seen = fill, seen
            else:
                self._val_cl_test_fill, self._val_cl_test_seen = fill, seen

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
            batch_index_n: Integer batch indices ``[N]``.
            continuous_covariates_nc: Continuous covariates ``[N, C]``.
            categorical_covariate_index_nd: Integer categorical covariate codes ``[N, D]``.

        Returns:
            A dictionary with keys:

            - ``"x_ng"``: Latent embeddings ``[N, n_latent]`` (keyed ``"x_ng"`` for
              pipeline compatibility).
            - ``"cell_type_probs_nc"``: Soft cell-type probabilities ``[N, n_classes]``.

            When :attr:`reconstruct_counts_on_predict` is ``True`` the dict contains only
            the reconstructed counts under ``"x_ng"`` (parent behaviour).
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
        logits_nc = self.cell_type_classifier(qz.mean)
        probs_nc = F.softmax(logits_nc, dim=-1)

        return {
            "x_ng": z_nk,
            "cell_type_probs_nc": probs_nc,
        }
