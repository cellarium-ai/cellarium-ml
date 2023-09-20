# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from cellarium.ml.data import DistributedAnnDataCollectionDataModule
from cellarium.ml.train import TrainingPlan


class _LightningCLIWithLinks(LightningCLI):
    """LightningCLI with custom argument linking."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("data.n_obs", "model.module.init_args.n_cells", apply_on="instantiate")
        parser.link_arguments("data.n_vars", "model.module.init_args.g_genes", apply_on="instantiate")
        parser.link_arguments("data.n_cell_types", "model.module.init_args.k_cell_types", apply_on="instantiate")


def main():
    _LightningCLIWithLinks(
        TrainingPlan,
        DistributedAnnDataCollectionDataModule,
    )


if __name__ == "__main__":
    main()
