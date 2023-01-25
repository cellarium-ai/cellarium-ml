from anndata import AnnData


class AnnDataSchema:
    """
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
        >>> schema.validate_adata(adata)

    Args:
        adata (AnnData): Reference AnnData.
    """

    attrs = ["obs", "obsm", "var", "varm", "var_names", "layers"]

    def __init__(self, adata: AnnData) -> None:
        self.attr_values = {}
        for attr in self.attrs:
            self.attr_values[attr] = getattr(adata, attr)

    def validate_anndata(self, adata: AnnData) -> None:
        """Validate anndata has proper attributes."""

        for attr in self.attrs:
            value = getattr(adata, attr)
            value_ref = self.attr_values[attr]
            if attr == "obs":
                # compare the elements inside the Index object and their order
                if not value_ref.columns.equals(value.columns):
                    raise ValueError(
                        ".obs attribute columns for anndata passed in does not match .obs attribute columns "
                        "of the reference anndata."
                    )
            elif attr in ["var", "var_names"]:
                # For var compare if two DataFrames have the same shape and elements
                # and the same row/column index.
                # For var_names compare the elements inside the Index object and their order
                if not value_ref.equals(value):
                    raise ValueError(
                        f".{attr} attribute for anndata passed in does not match .{attr} attribute "
                        "of the reference anndata."
                    )
            elif attr in ["layers", "obsm", "varm"]:
                # compare the keys
                # TODO: stricter comparison for varm
                if not set(value_ref.keys()) == set(value.keys()):
                    raise ValueError(
                        f".{attr} attribute keys for anndata passed in does not match .{attr} attribute keys "
                        "of the reference anndata."
                    )
