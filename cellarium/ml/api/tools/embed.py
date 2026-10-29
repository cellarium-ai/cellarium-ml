# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import ExitStack
from typing import Literal

import numpy as np
import torch

from cellarium.ml.api._obsm_storage import ObsmWriter
from cellarium.ml.api.cellariumdata import CellariumData, _device_of
from cellarium.ml.api.utils import resolve_device
from cellarium.ml.models.model import PredictMixin
from cellarium.ml.utilities.core import call_func_with_batch


@torch.no_grad()
def embed(
    cdata: CellariumData,
    module_key: str,
    key_added: str | None = None,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
    max_gb: float | None = None,
    batch_size: int | None = None,
) -> None:
    """
    Embed every cell of ``cdata`` with a trained module (for example PCA or scVI), in one pass over the cells, and store
    the embedding as ``cdata.obsm[key_added]``, an array of shape ``(n_obs, n_dimensions)`` in the order of the cells.

    This is for keeping the embedding of all the cells, so that later steps (such as
    :func:`~cellarium.ml.api.tools.propagate_labels`) can use it without reading the counts again. The array is in
    memory if it fits in ``max_gb`` (by default the ``max_gb`` of ``cdata``) and otherwise in a read-only
    memory-mapped file in the temporary directory, which is deleted when the array is no longer used. A view of
    ``cdata`` copies the rows of its cells into memory if they fit, and otherwise reads them from the file as needed.
    If the pass is interrupted, nothing is stored.

    Any data the module needs to run (such as scVI's batch column) is provided for you, see
    :meth:`CellariumData.using`. The module is the one in ``cdata.trained_modules[module_key]``, so for a view of cells
    (which starts with no trained modules) it is one that was trained on the view, or assigned to it.

    Args:
        cdata: The cells to embed (a :class:`CellariumData` or a view). It is where the embedding is stored.
        module_key: The key in ``cdata.trained_modules`` of the module to embed with. It must be a model that predicts
            an embedding, as PCA and scVI do.
        key_added: The key in ``obsm`` to store the embedding under, replacing any entry there. By default
            ``"X_<module_key>"`` (for example ``"X_pca"``), the name that :meth:`CellariumData.to_anndata` gives the
            embedding in the AnnData.
        accelerator: The accelerator for the pass, in ["cpu", "mps", "cuda", "auto"].
        max_gb: The most memory (in GiB) for the embedding, by default the ``max_gb`` of ``cdata``.
        batch_size: The number of cells per batch, by default that of the datamodule.
    """
    key_added = f"X_{module_key}" if key_added is None else key_added
    device = resolve_device(accelerator)
    stack = ExitStack()
    writer: ObsmWriter | None = None
    with stack, cdata.using(module_key) as trained:
        module = trained.module
        module.configure_model()  # idempotent; makes its transforms and model accessible
        model = module.model
        if not isinstance(model, PredictMixin):
            raise TypeError(f"The module '{module_key}' ({type(model).__name__}) cannot predict, so it cannot embed.")
        original_device = _device_of(model)
        model.to(device)
        try:
            for batch in cdata.predict_batches(
                module, device=device, batch_size=batch_size, desc=f"Embedding cells with '{module_key}'"
            ):
                prediction = call_func_with_batch(model.predict, batch)
                values = torch.as_tensor(prediction["x_ng"]).detach().cpu().numpy()
                positions = batch["row_indices_n"].numpy()
                if len(values) != len(positions):
                    raise RuntimeError(
                        f"The module '{module_key}' made {len(values)} embeddings for {len(positions)} cells."
                    )
                if writer is None:
                    writer = stack.enter_context(
                        ObsmWriter(cdata.n_obs, values.shape[1], cdata._max_bytes(max_gb), dtype=np.float32)
                    )
                if len(positions) > 1 and np.all(np.diff(positions) == 1):  # a contiguous run of cells
                    writer.array[positions[0] : positions[-1] + 1] = values
                else:
                    writer.array[positions] = values
        finally:
            if original_device is not None:
                model.to(original_device)
    assert writer is not None, "predict_batches yielded no batches"
    cdata.obsm[key_added] = writer.result
    cdata._embedding_sources[key_added] = (trained, writer.result)


def ensure_embedding(
    cdata: CellariumData,
    rep: str,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
    max_gb: float | None = None,
) -> str:
    """
    The key in ``cdata.obsm`` of the embedding of all the cells that ``rep`` names, making it if need be:

    * if ``rep`` is a key of ``cdata.obsm``, that is the embedding;
    * if ``rep`` is a key of ``cdata.trained_modules``, it is the embedding of that module, ``obsm["X_<rep>"]``, which
      is made with :func:`embed` unless ``cdata`` already has it from the module that is there now (an embedding of an
      earlier module, for example one inherited by a view that has been given a new module since, is made again).
    """
    if rep in cdata.obsm:
        return rep
    if rep not in cdata.trained_modules:
        raise KeyError(
            f"'{rep}' is neither a key of obsm ({list(cdata.obsm)}) nor of trained_modules "
            f"({list(cdata.trained_modules)}). Make an embedding of all the cells with cml.tl.embed(cdata, module_key)."
        )
    key = f"X_{rep}"
    state = cdata._cached_embedding_state(rep)
    if key in cdata.obsm and state is None:
        raise ValueError(
            f"obsm['{key}'] exists and was not made by cml.tl.embed from the module '{rep}', so it is not replaced by "
            f"the embedding of that module. Pass rep='{key}' to use it, or embed with another key_added."
        )
    if state != "current":
        embed(cdata, rep, key_added=key, accelerator=accelerator, max_gb=max_gb)
    return key
