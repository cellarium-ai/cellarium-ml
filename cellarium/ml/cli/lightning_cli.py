# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from cellarium.ml.data import DistributedAnnDataCollectionDataModule
from cellarium.ml.train.training_plan import TrainingPlan


def lightning_cli_factory(
    model: str,
    link_arguments: list[tuple[str, str]] | None = None,
    trainer_defaults: dict[str, Any] | None = None,
) -> LightningCLI:
    """
    Factory function for creating a LightningCLI with a preset model and custom argument linking.
    """

    class NewLightningCLI(LightningCLI):
        def __init__(self, args) -> None:
            super().__init__(
                TrainingPlan,
                DistributedAnnDataCollectionDataModule,
                trainer_defaults=trainer_defaults,
                args=args,
            )

        def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
            if link_arguments is not None:
                for arg1, arg2 in link_arguments:
                    parser.link_arguments(arg1, arg2, apply_on="instantiate")
            parser.set_defaults({"model.module": model})

    return NewLightningCLI
