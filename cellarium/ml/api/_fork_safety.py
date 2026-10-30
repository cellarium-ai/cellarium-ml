# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
A deltacells dataset reads tiles with a pool of threads, which it starts when it first reads one. A process that is
forked after that (a dataloader worker on Linux) inherits the pool but not its threads, so a tile that it asks the pool
to fetch is never fetched and the worker waits forever. The dataset can be pickled (the pool is left out, and made
again when it is first needed): the forked process does the same to the dataset that it inherits, with the dataset's own
``__getstate__`` and ``__setstate__``.
"""

import os
import weakref
from typing import Any

from cellarium.ml.api._view_collections import unwrap_collection
from cellarium.ml.data import DistributedDeltaCellsCollection

_COLLECTIONS: "weakref.WeakSet[DistributedDeltaCellsCollection]" = weakref.WeakSet()


def _reset_in_forked_process() -> None:
    for collection in list(_COLLECTIONS):
        dataset = collection.dataset
        dataset.__setstate__(dataset.__getstate__())


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_in_forked_process)


def reset_deltacells_after_fork(collection: Any) -> None:
    """Make forked processes reset the thread pool of ``collection`` (the collection it is a view of, if it is a view
    of a deltacells dataset), as above. Does nothing for other collections."""
    root = unwrap_collection(collection)
    if isinstance(root, DistributedDeltaCellsCollection):
        _COLLECTIONS.add(root)
