# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import numpy as np
from anndata import AnnData


class AnnDataSchema:
    r"""
    Store reference AnnData attributes for a collection of distributed AnnData objects.

    Validate AnnData objects against reference attributes.

    Example::

        >>> ref_adata = AnnData(X=np.zeros(2, 3))
        >>> ref_adata.obs["batch"] = [0, 1]
        >>> ref_adata.var["mu"] = ["a", "b", "c"]
        >>> adata = AnnData(X=np.zeros(4, 3))
        >>> adata.obs["batch"] = [2, 3, 4, 5]
        >>> adata.var["mu"] = ["a", "b", "c"]
        >>> schema = AnnDataSchema(ref_adata)
        >>> schema.validate_anndata(adata)

    Args:
        adata:
            Reference AnnData object.
        obs_columns_to_validate:
            Subset of columns to validate in the ``.obs`` attribute.
            If ``None``, all columns are validated.
    """

    attrs = ["obs", "obsm", "var", "varm", "varp", "var_names", "layers"]

    def __init__(self, adata: AnnData, obs_columns_to_validate: Sequence[str] | None = None) -> None:
        self.attr_values = {}
        for attr in self.attrs:
            # FIXME: some of the attributes have a reference to the anndata object itself.
            # This results in anndata object not being garbage collected.
            self.attr_values[attr] = getattr(adata, attr)
        self.obs_columns_to_validate = obs_columns_to_validate

    def validate_anndata(self, adata: AnnData) -> None:
        """Validate anndata has proper attributes."""

        for attr in self.attrs:
            value = getattr(adata, attr)
            ref_value = self.attr_values[attr]
            if attr == "obs":
                if self.obs_columns_to_validate is not None:
                    # Subset the columns to validate
                    ref_value = ref_value[self.obs_columns_to_validate]
                    value = value[self.obs_columns_to_validate]
                # compare the elements inside the Index object and their order
                if not ref_value.columns.equals(value.columns):
                    raise ValueError(
                        ".obs attribute columns for anndata passed in does not match .obs attribute columns "
                        "of the reference anndata."
                    )
                if not ref_value.dtypes.equals(value.dtypes):
                    raise ValueError(
                        ".obs attribute dtypes for anndata passed in does not match .obs attribute dtypes "
                        "of the reference anndata."
                    )
            elif attr in ["var", "var_names"]:
                # For var compare if two DataFrames have the same shape and elements
                # and the same row/column index.
                # For var_names compare the elements inside the Index object and their order
                if not ref_value.equals(value):
                    raise ValueError(
                        f".{attr} attribute for anndata passed in does not match .{attr} attribute "
                        "of the reference anndata."
                    )
            elif attr in ["layers", "obsm", "varm", "varp"]:
                # compare the keys
                if not set(ref_value.keys()) == set(value.keys()):
                    raise ValueError(
                        f".{attr} attribute keys for anndata passed in does not match .{attr} attribute keys "
                        "of the reference anndata."
                    )
                if attr in ["varm", "varp"]:
                    for key in ref_value:
                        arr = value[key]
                        ref_arr = ref_value[key]
                        if not np.array_equal(ref_arr, arr):
                            raise ValueError(
                                f".{attr} attribute for anndata passed in does not match .{attr} attribute "
                                "of the reference anndata."
                            )
