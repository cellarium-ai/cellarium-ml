# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from cellarium.ml.api.plotting import embedding  # noqa: E402


@pytest.fixture
def adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    n = 50
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "n_genes": rng.integers(100, 5000, n),
            "score": np.where(np.arange(n) % 10 == 0, np.nan, rng.random(n)),
        },
        index=[f"cell{i}" for i in range(n)],
    )
    result = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32), obs=obs)
    result.obsm["X_umap"] = rng.normal(size=(n, 2))
    return result


def test_categorical_color_has_the_category_as_the_tooltip_title(adata):
    fig = embedding(adata, color="cell_type", hover_data=["n_genes"])
    (trace,) = fig.data
    assert trace.hovertemplate == "<b>%{customdata[0]}</b><br>n_genes: %{customdata[1]}<extra></extra>"
    assert list(trace.customdata[:, 0]) == adata.obs["cell_type"].astype(str).tolist()
    assert "cell0" not in str(trace.customdata)  # the cell names are not shown
    assert not fig.layout.showlegend


def test_many_categories_get_distinct_colors_that_repeat(adata):
    adata.obs["many"] = pd.Categorical([f"type{i}" for i in range(len(adata))])
    colors = embedding(adata, color="many").data[0].marker.color
    assert len(set(colors)) > 20 and "#D3D3D3" not in colors


def test_continuous_color_draws_missing_values_in_gray(adata):
    fig = embedding(adata, color="score")
    assert len(fig.data) == 2 and fig.data[0].marker.color == "#D3D3D3"
    assert fig.data[1].marker.showscale


def test_no_tooltip_without_color_or_hover_data(adata):
    assert embedding(adata).data[0].hoverinfo == "skip"


def test_missing_basis_and_column_raise(adata):
    with pytest.raises(KeyError, match="pca"):
        embedding(adata, basis="pca")
    with pytest.raises(KeyError, match="nope"):
        embedding(adata, color="nope")


def test_figure_size(adata):
    layout = embedding(adata, width=500, height=300).layout
    assert (layout.width, layout.height) == (500, 300)
    assert embedding(adata).layout.height == 700


def test_save_html_writes_a_self_contained_file(adata, tmp_path):
    path = tmp_path / "umap.html"
    fig = embedding(adata, color="cell_type", save_html=str(path))
    text = path.read_text()
    assert "<html" in text[:200].lower()
    assert 'src="https://cdn.plot.ly' not in text and len(text) > 1_000_000  # plotly.js is embedded, not linked
    assert fig.data[0].marker.color is not None  # the figure is still returned


def test_save_html_accepts_a_path_object(adata, tmp_path):
    path = tmp_path / "UMAP.HTML"
    embedding(adata, save_html=path)
    assert path.exists()


def test_save_html_needs_an_html_suffix(adata, tmp_path):
    with pytest.raises(ValueError, match=r"\.html"):
        embedding(adata, save_html=tmp_path / "umap.png")
    with pytest.raises(ValueError, match=r"\.html"):
        embedding(adata, save_html=tmp_path / "umap")
    assert not list(tmp_path.iterdir())


def test_save_html_needs_an_existing_directory(adata, tmp_path):
    with pytest.raises(FileNotFoundError, match="directory"):
        embedding(adata, save_html=tmp_path / "missing" / "umap.html")


def test_no_file_is_written_by_default(adata, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    embedding(adata)
    assert not list(tmp_path.iterdir())
