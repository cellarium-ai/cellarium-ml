# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ShardCollection(Protocol):
    """
    Protocol satisfied by both :class:`~cellarium.ml.data.DistributedAnnDataCollection` and
    :class:`~cellarium.ml.data.DistributedArrowDataCollection`.

    Implementations must expose ``n_obs``, ``n_vars``, ``limits``, ``__getitem__``, ``__len__``,
    and :meth:`get_schema_metadata`.
    """

    limits: list[int]

    @property
    def n_obs(self) -> int:
        """Total number of observations across all shards."""
        ...

    @property
    def n_vars(self) -> int:
        """Number of variables (features)."""
        ...

    def __getitem__(self, idx: int | list[int]) -> Any:
        """Return data at the given index (or indices)."""
        ...

    def __len__(self) -> int:
        """Return the total number of observations."""
        ...

    def get_schema_metadata(self) -> dict[str, Any]:
        """
        Return schema-level metadata without loading any cell data.

        The returned dict contains at minimum:

        * ``"n_obs"`` – total number of observations (:class:`int`)
        * ``"n_vars"`` – number of variables (:class:`int`)
        * ``"var_names_g"`` – 1-D :class:`numpy.ndarray` of variable names (str)
        * ``"<key>_categories"`` – 1-D :class:`numpy.ndarray` of category strings for
          each categorical field present in the collection
        """
        ...
