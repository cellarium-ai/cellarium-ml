# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Command line interface for Cellarium ML.
"""

import copy
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from operator import attrgetter
from typing import Any

import numpy as np
import torch
import yaml
from jsonargparse import Namespace, class_from_function
from jsonargparse._loaders_dumpers import DefaultLoader
from jsonargparse._util import import_object
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule, CellariumPipeline
from cellarium.ml.utilities.data import collate_fn

cached_loaders = {}


@dataclass
class FileLoader:
    """
    A YAML constructor for loading a file and accessing its attributes.

    Example:

    .. code-block:: yaml

        model:
          transforms:
            - class_path: cellarium.ml.transforms.Filter
              init_args:
                filter_list:
                  !FileLoader
                  file_path: gs://dsp-cellarium-cas-public/test-data/filter_list.csv
                  loader_fn: pandas.read_csv
                  attr: index
                  convert_fn: numpy.ndarray.tolist

    Args:
        file_path:
            The file path to load the object from.
        loader_fn:
            A function to load the object from the file path.
        attr:
            An attribute to get from the loaded object. If ``None`` the loaded object is returned.
        convert_fn:
            A function to convert the loaded object. If ``None`` the loaded object is returned.
    """

    file_path: str
    loader_fn: Callable[[str], Any] | str
    attr: str | None = None
    convert_fn: Callable[[Any], Any] | str | None = None

    def __new__(cls, file_path, loader_fn, attr, convert_fn):
        if isinstance(loader_fn, str):
            loader_fn = import_object(loader_fn)
        if loader_fn not in cached_loaders:
            cached_loaders[loader_fn] = cache(loader_fn)
        loader_fn = cached_loaders[loader_fn]
        obj = loader_fn(file_path)

        if attr is not None:
            obj = attrgetter(attr)(obj)

        if isinstance(convert_fn, str):
            convert_fn = import_object(convert_fn)
        if convert_fn is not None:
            obj = convert_fn(obj)

        return obj


@dataclass
class CheckpointLoader(FileLoader):
    """
    A YAML constructor for loading a :class:`~cellarium.ml.core.CellariumModule` checkpoint and accessing its
    attributes.

    Example:

    .. code-block:: yaml

        model:
          transorms:
            - class_path: cellarium.ml.transforms.DivideByScale
              init_args:
                scale_g:
                  !CheckpointLoader
                  file_path: gs://dsp-cellarium-cas-public/test-data/tdigest.ckpt
                  attr: model.median_g
                  convert_fn: null

    Args:
        file_path:
            The file path to load the object from.
        attr:
            An attribute to get from the loaded object. If ``None`` the loaded object is returned.
        convert_fn:
            A function to convert the loaded object. If ``None`` the loaded object is returned.
    """

    file_path: str
    attr: str | None = None
    convert_fn: Callable[[Any], Any] | str | None = None

    def __new__(cls, file_path, attr, convert_fn):
        return super().__new__(cls, file_path, CellariumModule.load_from_checkpoint, attr, convert_fn)


def file_loader_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> FileLoader:
    """Construct an object from a file."""
    return FileLoader(**loader.construct_mapping(node))  # type: ignore[misc]


def checkpoint_loader_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> CheckpointLoader:
    """Construct an object from a checkpoint."""
    return CheckpointLoader(**loader.construct_mapping(node))  # type: ignore[misc]


loader = DefaultLoader
loader.add_constructor("!FileLoader", file_loader_constructor)
loader.add_constructor("!CheckpointLoader", checkpoint_loader_constructor)


REGISTERED_MODELS = {}


def register_model(model: Callable[[ArgsType], None]):
    REGISTERED_MODELS[model.__name__] = model
    return model


CellariumModuleLoadFromCheckpoint = class_from_function(CellariumModule.load_from_checkpoint, CellariumModule)


@dataclass
class LinkArguments:
    """
    Arguments for linking the value of a target argument to the values of one or more source arguments.

    Args:
        source:
            Key(s) from which the target value is derived.
        target:
            Key to where the value is set.
        compute_fn:
            Function to compute target value from source.
        apply_on:
            At what point to set target value, ``"parse"`` or ``"instantiate"``.
    """

    source: str | tuple[str, ...]
    target: str
    compute_fn: Callable | None = None
    apply_on: str = "instantiate"


def compute_n_obs(data: CellariumAnnDataDataModule) -> int:
    """
    Compute the number of observations in the data.

    Args:
        data: A :class:`CellariumAnnDataDataModule` instance.

    Returns:
        The number of observations in the data.
    """
    return data.dadc.n_obs


def compute_n_categories(data: CellariumAnnDataDataModule) -> int:
    """
    Compute the number of categories in the target variable.

    E.g. if the target variable is ``obs["cell_type"]`` then this function
    returns the number of categories in ``obs["cell_type"]``::

        >>> len(data.dadc[0].obs["cell_type"].cat.categories)

    Args:
        data: A :class:`CellariumAnnDataDataModule` instance.

    Returns:
        The number of categories in the target variable.
    """
    field = data.batch_keys["y_n"]
    value = getattr(data.dadc[0], field.attr)
    if field.key is not None:
        value = value[field.key]
    return len(value.cat.categories)


def compute_var_names_g(transforms: list[torch.nn.Module], data: CellariumAnnDataDataModule) -> np.ndarray:
    """
    Compute variable names from the data by applying the transforms.

    Args:
        transforms:
            A list of transforms.
        data:
            A :class:`CellariumAnnDataDataModule` instance.

    Returns:
        The variable names.
    """
    batch = {key: field(data.dadc, 0) for key, field in data.batch_keys.items()}
    pipeline = CellariumPipeline(transforms)
    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_batch = collate_fn([batch])
        with FakeCopyMode(fake_mode):
            fake_pipeline = copy.deepcopy(pipeline)
        fake_pipeline.to("cpu")
        output = fake_pipeline(fake_batch)
    return output["var_names_g"]


def lightning_cli_factory(
    model_class_path: str,
    link_arguments: list[LinkArguments] | None = None,
    trainer_defaults: dict[str, Any] | None = None,
) -> type[LightningCLI]:
    """
    Factory function for creating a :class:`LightningCLI` with a preset model and custom argument linking.

    Example::

        cli = lightning_cli_factory(
            "cellarium.ml.models.IncrementalPCA",
            link_arguments=[
                LinkArguments(("model.transforms", "data"), "model.model.init_args.var_names_g", compute_var_names_g)
            ],
            trainer_defaults={
                "max_epochs": 1,  # one pass
                "strategy": {
                    "class_path": "lightning.pytorch.strategies.DDPStrategy",
                    "dict_kwargs": {"broadcast_buffers": False},
                },
            },
        )

    Args:
        model_class_path:
            A string representation of the model class path (e.g., ``"cellarium.ml.models.IncrementalPCA"``).
        link_arguments:
            A list of :class:`LinkArguments` that specify how to derive the value of a target
            argument from the values of one or more source arguments. If ``None`` then no
            arguments are linked.
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
            with torch.device("meta"):
                # skip the initialization of model parameters
                # parameters are later initialized by the  `CellariumModule.configure_model` method
                return super().instantiate_classes()

        def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
            if link_arguments is not None:
                for link in link_arguments:
                    parser.link_arguments(link.source, link.target, link.compute_fn, link.apply_on)
            # this is helpful for generating a default config file with --print_config
            parser.set_defaults(
                {
                    "model.model": model_class_path,
                    "data.dadc": "cellarium.ml.data.DistributedAnnDataCollection",
                }
            )

    return NewLightningCLI


@register_model
def geneformer(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.Geneformer` model.

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
        "cellarium.ml.models.Geneformer",
        link_arguments=[
            LinkArguments(("model.transforms", "data"), "model.model.init_args.var_names_g", compute_var_names_g)
        ],
    )
    cli(args=args)


@register_model
def incremental_pca(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.IncrementalPCA` model.

    This example shows how to fit feature count data to incremental PCA
    model [1, 2].

    Example run::

        cellarium-ml incremental_pca fit \
            --model.model.init_args.n_components 50 \
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
        "cellarium.ml.models.IncrementalPCA",
        link_arguments=[
            LinkArguments(("model.transforms", "data"), "model.model.init_args.var_names_g", compute_var_names_g)
        ],
        trainer_defaults={
            "max_epochs": 1,  # one pass
            "strategy": {
                "class_path": "lightning.pytorch.strategies.DDPStrategy",
                "dict_kwargs": {"broadcast_buffers": False},
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
            --data.batch_keys.var_names_g.attr var_names \
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

    cli = lightning_cli_factory(
        "cellarium.ml.models.LogisticRegression",
        link_arguments=[
            LinkArguments(("model.transforms", "data"), "model.model.init_args.var_names_g", compute_var_names_g),
            LinkArguments("data", "model.model.init_args.n_obs", compute_n_obs),
            LinkArguments("data", "model.model.init_args.n_categories", compute_n_categories),
        ],
    )
    cli(args=args)


@register_model
def onepass_mean_var_std(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.OnePassMeanVarStd` model.

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
        "cellarium.ml.models.OnePassMeanVarStd",
        link_arguments=[
            LinkArguments(("model.transforms", "data"), "model.model.init_args.var_names_g", compute_var_names_g)
        ],
        trainer_defaults={
            "max_epochs": 1,  # one pass
            "strategy": {
                "class_path": "lightning.pytorch.strategies.DDPStrategy",
                "dict_kwargs": {"broadcast_buffers": False},
            },
        },
    )
    cli(args=args)


@register_model
def probabilistic_pca(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.ProbabilisticPCA` model.

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
            --model.model.init_args.n_components 256 \
            --model.model.init_args.ppca_flavor marginalized \
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

    **References:**

    1. `Probabilistic Principal Component Analysis (Tipping et al.)
       <https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf>`_.
    2. `Understanding Posterior Collapse in Generative Latent Variable Models (Lucas et al.)
       <https://openreview.net/pdf?id=r1xaVLUYuE>`_.

    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """
    cli = lightning_cli_factory(
        "cellarium.ml.models.ProbabilisticPCA",
        link_arguments=[
            LinkArguments(("model.transforms", "data"), "model.model.init_args.var_names_g", compute_var_names_g),
            LinkArguments("data", "model.model.init_args.n_obs", compute_n_obs),
        ],
    )
    cli(args=args)


@register_model
def tdigest(args: ArgsType = None) -> None:
    r"""
    CLI to run the :class:`cellarium.ml.models.TDigest` model.

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
        "cellarium.ml.models.TDigest",
        link_arguments=[
            LinkArguments(("model.transforms", "data"), "model.model.init_args.var_names_g", compute_var_names_g)
        ],
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
        if "model_name" not in args:
            raise ValueError("'model_name' key must be specified in args")
        model_name = args.pop("model_name")
    elif isinstance(args, list):
        if len(args) == 0:
            raise ValueError("'model_name' must be specified as the first argument in args")
        model_name = args.pop(0)
    elif args is None:
        args = sys.argv[1:].copy()
        if len(args) == 0:
            raise ValueError("'model_name' must be specified after cellarium-ml")
        model_name = args.pop(0)

    if model_name not in REGISTERED_MODELS:
        raise ValueError(f"'model_name' must be one of {list(REGISTERED_MODELS.keys())}. Got '{model_name}'")
    model_cli = REGISTERED_MODELS[model_name]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Transforming to str index.")
        warnings.filterwarnings("ignore", message="LightningCLI's args parameter is intended to run from within Python")
        warnings.filterwarnings("ignore", message="Your `IterableDataset` has `__len__` defined.")
        model_cli(args)  # run the model


if __name__ == "__main__":
    main()
