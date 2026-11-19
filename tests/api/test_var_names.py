# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pytest

from cellarium.ml.api import CellariumData
from cellarium.ml.api.cellariumdata import _strip_ensembl_version
from cellarium.ml.api.preprocessing import highly_variable_genes, pseudobulk

N_GENES = 30
# every gene name occurs twice, so the names are not unique
GENE_NAMES = [f"name{k // 2}" for k in range(N_GENES)]
# unique Ensembl IDs with versions, unique without them too
VERSIONED_IDS = [f"ENSG{k:011d}.{k % 3 + 1}" for k in range(N_GENES)]
# unique Ensembl IDs with versions, which differ only in the version two by two, so are repeated without them
COLLIDING_IDS = [f"ENSG{k // 2:011d}.{k % 2 + 1}" for k in range(N_GENES)]


def suffixed(names: list[str]) -> list[str]:
    """``names`` (each occurring at most twice) with the second copy of a name suffixed, in the order given."""
    seen: set[str] = set()
    return [f"{name}-1" if name in seen or seen.add(name) else name for name in names]  # type: ignore[func-returns-value]


@pytest.fixture(params=["h5ad", "deltacells"])
def make_cdata(request, make_h5ad_files, tmp_path, deltacells_kwargs):
    """
    A factory of CellariumData over files whose genes have the unique ``gene_id`` and the repeated ``gene_name``, and
    the versioned Ensembl IDs ``gene_id_versioned`` and ``gene_id_colliding`` (repeated without the versions).
    """
    h5ad_paths = make_h5ad_files(n_files=2, cells_per_file=20, n_genes=N_GENES)
    for path in h5ad_paths:
        adata = ad.read_h5ad(path)
        adata.var["gene_name"] = GENE_NAMES
        adata.var["gene_id"] = [f"ENSG{k:03d}" for k in range(N_GENES)]
        adata.var["gene_id_versioned"] = VERSIONED_IDS
        adata.var["gene_id_colliding"] = COLLIDING_IDS
        adata.write_h5ad(path)

    def make(**kwargs) -> CellariumData:
        if request.param == "h5ad":
            return CellariumData(h5ad_paths=h5ad_paths, **kwargs)
        pytest.importorskip("deltacells._core")
        from cellarium.ml.api import create_deltacells_dataset

        uri = create_deltacells_dataset(h5ad_paths, str(tmp_path / "deltacells"), tile_size=15, level=3, log=None)
        return CellariumData.from_deltacells(uri, deltacells_kwargs=deltacells_kwargs, **kwargs)

    return make


def test_unique_var_key_is_used_as_the_gene_names(make_cdata):
    cdata = make_cdata(var_key="gene_id")
    assert cdata.var.index.is_unique
    assert cdata.var.index.str.startswith("ENSG").all()
    assert "gene_id" not in cdata.var.columns


def test_repeated_var_key_raises_and_lists_the_unique_columns(make_cdata):
    with pytest.raises(ValueError, match="var_key='gene_name'.*15 of them are repeated") as error:
        make_cdata(var_key="gene_name")
    message = str(error.value)
    assert "'gene_id'" in message
    assert "var_names_make_unique=True" in message


def test_make_unique_suffixes_the_repeats_and_keeps_the_original_names(make_cdata):
    cdata = make_cdata(var_key="gene_name", var_names_make_unique=True)
    original = cdata.var["gene_name_original"].tolist()
    assert sorted(original) == sorted(GENE_NAMES)  # (deltacells stores the genes in its own order)
    assert cdata.var.index.tolist() == suffixed(original)
    assert cdata.datamodule.var_names_g.tolist() == suffixed(original)
    assert "gene_id" in cdata.var.columns


def test_make_unique_leaves_unique_names_unchanged(make_cdata):
    cdata = make_cdata(var_key="gene_id", var_names_make_unique=True)
    assert cdata.var.index.str.startswith("ENSG").all()
    assert cdata.var["gene_id_original"].tolist() == cdata.var.index.tolist()


def test_make_unique_is_carried_by_views(make_cdata):
    cdata = make_cdata(var_key="gene_name", var_names_make_unique=True)
    view = cdata[np.arange(0, 10)]
    assert view.var.index.tolist() == cdata.var.index.tolist()
    assert view.datamodule.var_names_g.tolist() == cdata.var.index.tolist()


@pytest.mark.parametrize("flavor", ["seurat", "seurat_v3", "kotliar"])
def test_highly_variable_genes_with_made_unique_names(make_cdata, flavor):
    cdata = make_cdata(var_key="gene_name", var_names_make_unique=True)
    hvg_df = highly_variable_genes(cdata, n_top_genes=10, flavor=flavor, accelerator="cpu")
    assert hvg_df.index.is_unique
    assert cdata.hvg is not None
    assert cdata.hvg.index.tolist() == cdata.var.index.tolist()
    assert int(cdata.hvg.sum()) == int(hvg_df["highly_variable"].sum())


def test_pseudobulk_has_the_made_unique_names(make_cdata):
    cdata = make_cdata(var_key="gene_name", var_names_make_unique=True)
    adata = pseudobulk(cdata, batch_key="cell_type", accelerator="cpu")
    assert adata.var_names.tolist() == cdata.var.index.tolist()
    assert adata.var["gene_name_original"].tolist() == cdata.var["gene_name_original"].tolist()


def test_strip_ensembl_version_strips_only_the_versions_of_ensembl_gene_ids():
    names = [
        "ENSG00000198804.2",
        "ENSG00000182378.15_PAR_Y",
        "ENSMUSG00000064370.1",
        "ENSG00000198804",
        "AC000061.1",
        "RP11-34P13.3",
        "MT-CO1",
        "ENST00000361390.2",
    ]
    assert _strip_ensembl_version(np.array(names)).tolist() == [
        "ENSG00000198804",
        "ENSG00000182378_PAR_Y",
        "ENSMUSG00000064370",
        "ENSG00000198804",
        "AC000061.1",
        "RP11-34P13.3",
        "MT-CO1",
        "ENST00000361390.2",
    ]


def test_strip_ensembl_version_strips_the_gene_names_and_keeps_the_original_names(make_cdata):
    cdata = make_cdata(var_key="gene_id_versioned", var_names_strip_ensembl_version=True)
    original = cdata.var["gene_id_versioned_original"].tolist()
    assert sorted(original) == sorted(VERSIONED_IDS)  # (deltacells stores the genes in its own order)
    assert cdata.var.index.tolist() == [name.split(".")[0] for name in original]
    assert cdata.datamodule.var_names_g.tolist() == cdata.var.index.tolist()
    view = cdata[np.arange(0, 10)]
    assert view.datamodule.var_names_g.tolist() == cdata.var.index.tolist()


def test_strip_ensembl_version_leaves_other_names_unchanged(make_cdata):
    cdata = make_cdata(var_key="gene_id", var_names_strip_ensembl_version=True)
    assert cdata.var.index.tolist() == cdata.var["gene_id_original"].tolist()


def test_names_repeated_once_the_versions_are_stripped_raise(make_cdata):
    with pytest.raises(ValueError, match="var_key='gene_id_colliding', with the Ensembl versions stripped") as error:
        make_cdata(var_key="gene_id_colliding", var_names_strip_ensembl_version=True)
    assert "var_names_make_unique=True" in str(error.value)


def test_strip_ensembl_version_then_make_unique(make_cdata):
    cdata = make_cdata(var_key="gene_id_colliding", var_names_strip_ensembl_version=True, var_names_make_unique=True)
    original = cdata.var["gene_id_colliding_original"].tolist()
    assert sorted(original) == sorted(COLLIDING_IDS)
    assert cdata.var.index.tolist() == suffixed([name.split(".")[0] for name in original])
    assert cdata.datamodule.var_names_g.tolist() == cdata.var.index.tolist()


def test_strip_ensembl_version_survives_save_and_load(make_cdata, tmp_path):
    cdata = make_cdata(var_key="gene_id_versioned", var_names_strip_ensembl_version=True)
    cdata.save(str(tmp_path / "session"))
    loaded = CellariumData.load(str(tmp_path / "session"))
    assert loaded.var.index.tolist() == cdata.var.index.tolist()
    assert loaded.datamodule.var_names_g.tolist() == cdata.var.index.tolist()
