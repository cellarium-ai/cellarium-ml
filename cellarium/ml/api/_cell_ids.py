# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Cell IDs: the integers that identify the cells of a :class:`~cellarium.ml.api.CellariumData` for all indexing.

The ID of a cell is its position in the data that was opened (the order of ``h5ad_paths``, or of the deltacells
dataset), so nothing needs to be stored for it. The one exception is a deltacells dataset written by
:meth:`~cellarium.ml.api.CellariumData.to_deltacells`: its cells are a subset of those of another dataset, so it stores
the IDs they have there, in the ``obs`` column :data:`CELL_ID_COLUMN`. When a dataset has that column it is used as the
IDs (and when it does not, which is the case for every source dataset, the IDs are the positions). This is what lets
labels computed on a compacted dataset, in another session, be applied to the cells of the data it was made from.
"""

CELL_ID_COLUMN = "cellarium_cell_id"
