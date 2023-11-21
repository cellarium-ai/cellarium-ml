# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Command line interface for Cellarium ML.
"""

import sys
import warnings
from collections.abc import Callable
from typing import Any

from jsonargparse import Namespace
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule

REGISTERED_MODELS = {}


def register_model(model: Callable[[ArgsType], None]):
    REGISTERED_MODELS[model.__name__] = model
    return model


def lightning_cli_factory(
    model_class_path: str,
    link_arguments: list[tuple[str, str, Callable[[Any], object] | None]] | None = None,
    trainer_defaults: dict[str, Any] | None = None,
) -> type[LightningCLI]:
    """
    Factory function for creating a :class:`LightningCLI` with a preset model and custom argument linking.

    Example::

        cli = lightning_cli_factory(
            "cellarium.ml.models.IncrementalPCAFromCLI",
            link_arguments=[("data.n_vars", "model.model.init_args.g_genes", None)],
            trainer_defaults={
                "max_epochs": 1,  # one pass
                "strategy": {
                    "class_path": "lightning.pytorch.strategies.DDPStrategy",
                    "init_args": {"broadcast_buffers": False},
                },
            },
        )

    Args:
        model_class_path:
            A string representation of the model class path (e.g., ``"cellarium.ml.models.IncrementalPCAFromCLI"``).
        link_arguments:
            A list of tuples of the form ``(source, target)`` where ``source`` is linked to ``target``.
        trainer_defaults:
            Default values for the trainer.

    Returns:
        A :class:`LightningCLI` class with the given model and argument linking.
    """

    class NewLightningCLI(LightningCLI):
        def __init__(self, args: ArgsType = None) -> None:
            super().__init__(
                CellariumModule,
                CellariumAnnDataDataModule,
                trainer_defaults=trainer_defaults,
                args=args,
            )

        def instantiate_classes(self) -> None:
            super().instantiate_classes()

            # impute linked arguments after instantiation
            if link_arguments is not None:
                for source, target, compute_fn in link_arguments:
                    # e.g., source == "data.var_names"
                    source_key, *source_attrs = source.split(".")
                    # note that config_init is initialized, so value is an instance
                    config_init = self.config_init[self.subcommand]
                    value = config_init[source_key]
                    for attr in source_attrs:
                        value = getattr(value, attr)
                    if compute_fn is not None:
                        value = compute_fn(value)

                    # e.g., target == "model.model.init_args.feature_schema"
                    target_keys = target.split(".")
                    # note that config is dict-like, so assign the value to the last key
                    config = self.config[self.subcommand]
                    for key in target_keys[:-1]:
                        config = config[key]
                    config[target_keys[-1]] = value

            # save the config to the :attr:`model.hparams` to be able to load the model checkpoint
            self.model._set_hparams(self.config[self.subcommand])

        def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
            if link_arguments is not None:
                for arg1, arg2, compute_fn in link_arguments:
                    parser.link_arguments(arg1, arg2, compute_fn=compute_fn, apply_on="instantiate")
            parser.set_defaults({"model.model": model_class_path})

    return NewLightningCLI


@register_model
def geneformer(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.GeneformerFromCLI` model.

    This example shows how to fit feature count data to the Geneformer model [1].

    Example run::

        cellarium-ml geneformer fit \
            --data.filenames "gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad" \
            --data.shard_size 100 \
            --data.max_cache_size 2 \
            --data.batch_size 5 \
            --data.num_workers 1 \
            --trainer.accelerator gpu \
            --trainer.devices 1 \
            --trainer.default_root_dir runs/geneformer \
            --trainer.max_steps 10

    **References:**

    1. `Transfer learning enables predictions in network biology (Theodoris et al.)
       <https://www.nature.com/articles/s41586-023-06139-9>`_.

    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """
    cli = lightning_cli_factory(
        "cellarium.ml.models.GeneformerFromCLI",
        link_arguments=[("data.var_names", "model.model.init_args.feature_schema", None)],
    )
    cli(args=args)


@register_model
def incremental_pca(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.IncrementalPCAFromCLI` model.

    This example shows how to fit feature count data to incremental PCA
    model [1, 2].

    Example run::

        cellarium-ml incremental_pca fit \
            --model.model.init_args.k_components 50 \
            --data.filenames "gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad" \
            --data.shard_size 100 \
            --data.max_cache_size 2 \
            --data.batch_size 100 \
            --data.num_workers 4 \
            --trainer.accelerator gpu \
            --trainer.devices 1 \
            --trainer.default_root_dir runs/ipca \

    **References:**

    1. `A Distributed and Incremental SVD Algorithm for Agglomerative Data Analysis on Large Networks (Iwen et al.)
       <https://users.math.msu.edu/users/iwenmark/Papers/distrib_inc_svd.pdf>`_.
    2. `Incremental Learning for Robust Visual Tracking (Ross et al.)
       <https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf>`_.

    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """
    cli = lightning_cli_factory(
        "cellarium.ml.models.IncrementalPCAFromCLI",
        link_arguments=[("data.n_vars", "model.model.init_args.g_genes", None)],
        trainer_defaults={
            "max_epochs": 1,  # one pass
            "strategy": {
                "class_path": "lightning.pytorch.strategies.DDPStrategy",
                "init_args": {"broadcast_buffers": False},
            },
        },
    )
    cli(args=args)


@register_model
def logistic_regression(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.LogisticRegression` model.

    Example run::

        cellarium-ml logistic_regression fit \
            --data.filenames "gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad" \
            --data.shard_size 100 \
            --data.max_cache_size 2 \
            --data.batch_keys.x_ng.attr X \
            --data.batch_keys.x_ng.convert_fn cellarium.ml.utilities.data.densify \
            --data.batch_keys.feature_g.attr var_names \
            --data.batch_keys.y_n.attr obs \
            --data.batch_keys.y_n.key cell_type \
            --data.batch_keys.y_n.convert_fn cellarium.ml.utilities.data.categories_to_codes \
            --data.batch_size 100 \
            --data.num_workers 4 \
            --trainer.accelerator gpu \
            --trainer.devices 1 \
            --trainer.max_steps 1000

    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """

    def get_c_categories(data: CellariumAnnDataDataModule) -> int:
        """
        Get the number of categories in the target variable.

        E.g. if the target variable is ``obs["cell_type"]`` then this function
        returns the number of categories in ``obs["cell_type"]``::

            >>> len(data.dadc[0].obs["cell_type"].cat.categories)
        """
        field = data.batch_keys["y_n"]
        value = getattr(data.dadc[0], field.attr)
        if field.key is not None:
            value = value[field.key]
        return len(value.cat.categories)

    cli = lightning_cli_factory(
        "cellarium.ml.models.LogisticRegression",
        link_arguments=[
            ("data.n_obs", "model.model.init_args.n_obs", None),
            ("data.var_names", "model.model.init_args.feature_schema", None),
            ("data", "model.model.init_args.c_categories", get_c_categories),
        ],
    )
    cli(args=args)


@register_model
def onepass_mean_var_std(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.OnePassMeanVarStdFromCLI` model.

    This example shows how to calculate mean, variance, and standard deviation of log normalized
    feature count data in one pass [1].

    Example run::

        cellarium-ml onepass_mean_var_std fit \
            --data.filenames "gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad" \
            --data.shard_size 100 \
            --data.max_cache_size 2 \
            --data.batch_size 100 \
            --data.num_workers 4 \
            --trainer.accelerator gpu \
            --trainer.devices 1 \
            --trainer.default_root_dir runs/onepass \

    **References:**

    1. `Algorithms for calculating variance
       <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.

    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """
    cli = lightning_cli_factory(
        "cellarium.ml.models.OnePassMeanVarStdFromCLI",
        link_arguments=[("data.n_vars", "model.model.init_args.g_genes", None)],
        trainer_defaults={
            "max_epochs": 1,  # one pass
            "strategy": {
                "class_path": "lightning.pytorch.strategies.DDPStrategy",
                "init_args": {"broadcast_buffers": False},
            },
        },
    )
    cli(args=args)


@register_model
def probabilistic_pca(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.ProbabilisticPCAFromCLI` model.

    This example shows how to fit feature count data to probabilistic PCA
    model [1].

    There are two flavors of probabilistic PCA model that are available:

    1. ``marginalized`` - latent variable ``z`` is marginalized out [1]. Marginalized
       model provides a closed-form solution for the marginal log-likelihood.
       Closed-form solution for the marginal log-likelihood has reduced
       variance compared to the ``linear_vae`` model.
    2. ``linear_vae`` - latent variable ``z`` has a diagonal Gaussian distribution [2].
       Training a linear VAE with variational inference recovers a uniquely identifiable
       global maximum  corresponding to the principal component directions.
       The global maximum of the ELBO objective for the linear VAE  is identical
       to the global maximum for the marginal log-likelihood of probabilistic PCA.

    Example run::

        cellarium-ml probabilistic_pca fit \
            --model.model.init_args.mean_var_std_ckpt_path \
            "runs/onepass/lightning_logs/version_0/checkpoints/module_checkpoint.pt" \
            --data.filenames "gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad" \
            --data.shard_size 100 \
            --data.max_cache_size 2 \
            --data.batch_size 100 \
            --data.shuffle true \
            --data.num_workers 4 \
            --trainer.accelerator gpu \
            --trainer.devices 1 \
            --trainer.max_steps 1000 \
            --trainer.default_root_dir runs/ppca \
            --trainer.callbacks cellarium.ml.callbacks.VarianceMonitor \
            --trainer.callbacks.mean_var_std_ckpt_path \
            "runs/onepass/lightning_logs/version_0/checkpoints/module_checkpoint.pt"

    **References:**

    1. `Probabilistic Principal Component Analysis (Tipping et al.)
       <https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf>`_.
    2. `Understanding Posterior Collapse in Generative Latent Variable Models (Lucas et al.)
       <https://openreview.net/pdf?id=r1xaVLUYuE>`_.

    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """
    cli = lightning_cli_factory(
        "cellarium.ml.models.ProbabilisticPCAFromCLI",
        link_arguments=[
            ("data.n_obs", "model.model.init_args.n_cells", None),
            ("data.n_vars", "model.model.init_args.g_genes", None),
        ],
    )
    cli(args=args)


@register_model
def tdigest(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.TDigestFromCLI` model.

    This example shows how to calculate non-zero median of normalized feature count
    data in one pass [1].

    Example run::

        cellarium-ml tdigest fit \
            --data.filenames "gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad" \
            --data.shard_size 100 \
            --data.max_cache_size 2 \
            --data.batch_size 100 \
            --data.num_workers 4 \
            --trainer.accelerator cpu \
            --trainer.devices 1 \
            --trainer.default_root_dir runs/tdigest \

    **References:**

    1. `Computing Extremely Accurate Quantiles Using T-Digests (Dunning et al.)
       <https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf>`_.

    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """
    cli = lightning_cli_factory(
        "cellarium.ml.models.TDigestFromCLI",
        link_arguments=[("data.n_vars", "model.model.init_args.g_genes", None)],
        trainer_defaults={
            "max_epochs": 1,  # one pass
        },
    )
    cli(args=args)


def main(args: ArgsType = None) -> None:
    """
    CLI that dispatches to the appropriate model cli based on the model name in ``args`` and runs it.

    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
            The model name is expected to be the first argument if ``args`` is a list
            or the ``model_name`` key if ``args`` is a dictionary or ``Namespace``.
    """
    if isinstance(args, (dict, Namespace)):
        assert "model_name" in args, "'model_name' key must be specified in args"
        model_name = args.pop("model_name")
    elif isinstance(args, list):
        assert len(args) > 0, "'model_name' must be specified as the first argument in args"
        model_name = args.pop(0)
    elif args is None:
        args = sys.argv[1:].copy()
        assert len(args) > 0, "'model_name' must be specified after cellarium-ml"
        model_name = args.pop(0)

    assert (
        model_name in REGISTERED_MODELS
    ), f"'model_name' must be one of {list(REGISTERED_MODELS.keys())}. Got '{model_name}'"
    model_cli = REGISTERED_MODELS[model_name]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightningCLI's args parameter is intended to run from within Python")
        warnings.filterwarnings("ignore", message="Your `IterableDataset` has `__len__` defined.")
        model_cli(args)  # run the model


if __name__ == "__main__":
    main()
