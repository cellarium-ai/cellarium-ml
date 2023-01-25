from anndata import AnnData


class AnnDataSchema:
    """
    Store reference AnnData attributes for a collection of distributed AnnData objects.

    Validate AnnData objects against reference attributes.
    """

    attrs = ["obs", "obsm", "layers", "var", "var_names"]

    def __init__(self, adata: AnnData) -> None:
        self.attr_values = {}
        for attr in self.attrs:
            self.attr_values[attr] = getattr(adata, attr)

    def validate_adata(self, adata: AnnData) -> None:
        for attr in self.attrs:
            value = getattr(adata, attr)
            value_ref = self.attr_values[attr]
            if attr == "obs":
                # compare the elements inside the Index object and their order
                if not value_ref.columns.equals(value.columns):
                    raise ValueError(
                        "AnnData object's .obs attribute must have the same "
                        "columns as the reference AnnData object's."
                    )
            elif attr == "var":
                # compare if two DataFrames have the same shape and elements
                # and the same row/column index.
                if not value_ref.equals(value):
                    raise ValueError(
                        "AnnData object's .var attribute must have the same "
                        "as the reference AnnData object's."
                    )
            elif attr == "var_names":
                # compare the elements inside the Index object and their order
                if not value_ref.equals(value):
                    raise ValueError(
                        "AnnData object's .var_names attribute must be the same "
                        "as the reference AnnData object's."
                    )
            elif attr in ["layers", "obsm"]:
                # compare the keys
                if not set(value_ref.keys()) == set(value.keys()):
                    raise ValueError(
                        f"AnnData object's .{attr} attribute must be the same "
                        "keys as the reference AnnData object's."
                    )
