# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch


def write_embedding(
    embedding: torch.Tensor,
    ids: torch.Tensor,
    output_dir: Path | str,
    postfix: int | str,
) -> None:
    """
    Write embeddings to a CSV file.

    Args:
        embedding: The embeddings to write.
        ids: The CAS IDs of the cells.
        output_dir: The directory to write the embeddings to.
        postfix: A postfix to add to the CSV file name.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(embedding.cpu())
    df.insert(0, "db_ids", ids.cpu())
    df.to_csv(f"{output_dir}/batch_{postfix}.csv", header=False, index=False)


class EmbeddingWriter(pl.callbacks.BasePredictionWriter):
    """
    Write embeddings to a CSV file. The CSV file will have the same number of rows as the
    number of embeddings, and the number of columns will be the same as the embedding size.
    The first column will be the CAS ID of each cell.

    Args:
        output_dir: The directory to write the embeddings to.
        embedding_size: The size of the embeddings. If ``None``, the entire embedding will be
            written. If not ``None``, only the first ``embedding_size`` columns will be written.
    """

    def __init__(
        self, output_dir: Path | str, embedding_size: int | None = None
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.embedding_size = embedding_size

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: torch.Tensor,
        batch_indices: list[int],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.embedding_size is not None:
            prediction = prediction[:, : self.embedding_size]

        write_embedding(
            embedding=prediction,
            ids=batch["obs_names"],
            output_dir=self.output_dir,
            postfix=batch_idx * trainer.world_size + trainer.global_rank,
        )
