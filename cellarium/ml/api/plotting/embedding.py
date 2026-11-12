# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import colorsys
import html
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from anndata import AnnData

if TYPE_CHECKING:
    import plotly.graph_objects as go

_NA_COLOR = "#D3D3D3"  # light gray, for cells with a missing value
_NO_COLOR = "#4C78A8"  # for the cells when there is nothing to color by
_FALSE_COLOR = "#D3D3D3"  # light gray, for False in a boolean column
_TRUE_COLOR = "#D62728"  # tab:red, for True in a boolean column
_CONTINUOUS_SCALE = "Oranges"
_CONTINUOUS_SCALE_START = 0.15  # Oranges starts almost white, which vanishes on a white background


def _qualitative_palette() -> list[str]:
    """
    A long list of distinguishable colors, for categoricals with many categories (the colors repeat beyond the end of
    it). It is made of the qualitative palettes of plotly, without the colors that are grayish (these would be
    confused with the color of missing values) or pale (these are hard to see on a white background).
    """
    import plotly.express as px

    names = ["Plotly", "D3", "T10", "G10", "Alphabet", "Dark24", "Light24", "Set1", "Bold", "Vivid", "Safe", "Dark2"]
    palette: list[str] = []
    for name in names:
        for color in getattr(px.colors.qualitative, name):
            color = _to_hex(color)
            if color not in palette and _is_usable(color):
                palette.append(color)
    return palette


def _to_hex(color: str) -> str:
    """``color`` (``#rrggbb`` or ``rgb(r, g, b)``) as upper case ``#RRGGBB``."""
    if color.startswith("#"):
        return color.upper()
    red, green, blue = (int(float(part)) for part in color[color.index("(") + 1 : color.index(")")].split(",")[:3])
    return f"#{red:02X}{green:02X}{blue:02X}"


def _is_usable(color: str) -> bool:
    red, green, blue = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    _, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    return saturation >= 0.35 and 0.2 <= lightness <= 0.7


def _find_basis(adata: AnnData, basis: str) -> str:
    """The key in ``adata.obsm`` of the embedding ``basis``: ``X_<basis>`` first, as in scanpy, and then ``basis``."""
    for key in (f"X_{basis}", basis):
        if key in adata.obsm:
            return key
    raise KeyError(
        f"There is no embedding '{basis}' in adata.obsm (looked for 'X_{basis}' and '{basis}'). "
        f"It has {list(adata.obsm.keys())}."
    )


def _check_obs_columns(adata: AnnData, columns: Sequence[str], argument: str) -> None:
    missing = [column for column in columns if column not in adata.obs.columns]
    if missing:
        raise KeyError(f"{argument} {missing} are not columns of adata.obs. It has {list(adata.obs.columns)}.")


def _format_values(values: pd.Series) -> np.ndarray:
    """The values of a column as the (html-escaped) text for the tooltip, with ``NA`` for the missing ones."""
    missing = values.isna().to_numpy()
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        as_float = values.to_numpy(dtype=float, na_value=np.nan)
        text = np.array(
            [str(int(v)) if np.isfinite(v) and v == int(v) and abs(v) < 1e15 else f"{v:.4g}" for v in as_float],
            dtype=object,
        )
    else:
        text = values.astype(str).to_numpy(dtype=object)
    text = np.array([html.escape(str(t)) for t in text], dtype=object)
    text[missing] = "NA"
    return text


def _categorical_colors(
    adata: AnnData,
    color: str,
    categories: pd.Series,
    palette: Sequence[str] | Mapping[str, str] | None,
) -> np.ndarray:
    """The color of each cell, for the categorical ``categories`` (the missing values are light gray)."""
    levels = list(categories.cat.categories)
    stored = adata.uns.get(f"{color}_colors")
    if palette is None and stored is not None and len(stored) == len(levels):
        palette = list(stored)  # the colors that scanpy stored for the categories
    if isinstance(palette, Mapping):
        missing = [level for level in levels if level not in palette]
        if missing:
            raise ValueError(f"palette has no color for the categories {missing[:5]} of '{color}'.")
        level_colors = [palette[level] for level in levels]
    else:
        colors = list(palette) if palette is not None else _qualitative_palette()
        if len(colors) == 0:
            raise ValueError("palette is empty.")
        level_colors = [colors[i % len(colors)] for i in range(len(levels))]  # repeats if there are too many categories
    codes = categories.cat.codes.to_numpy()
    lookup = np.array([*level_colors, _NA_COLOR], dtype=object)
    return lookup[np.where(codes < 0, len(levels), codes)]


def embedding(
    adata: AnnData,
    basis: str = "umap",
    color: str | None = None,
    hover_data: str | Sequence[str] | None = None,
    components: tuple[int, int] = (0, 1),
    palette: Sequence[str] | Mapping[str, str] | None = None,
    size: float | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = 700,
    show: bool = False,
    save_html: str | os.PathLike[str] | None = None,
) -> "go.Figure":
    """
    An interactive, zoomable scatter plot of an embedding of the cells (for example a UMAP), in the manner of
    :func:`scanpy.pl.embedding` but as a `plotly <https://plotly.com/python/>`_ figure that can be zoomed and panned,
    and whose cells show a tooltip with their metadata when hovered. It is meant for exploring (for example a sketch of
    a large dataset): there is no legend, because a categorical, such as a cell type, may have hundreds of categories.

    The tooltip of a cell is titled with its value of ``color`` (in bold) and lists its values of the columns in
    ``hover_data`` below it. The name of the cell is not shown. If there is no ``color`` and no ``hover_data`` there is
    no tooltip.

    This needs plotly, which is not a dependency of cellarium-ml: ``pip install plotly``.

    Args:
        adata: The cells. The embedding is in ``adata.obsm`` and the metadata of the cells in ``adata.obs``.
        basis: The embedding to plot, ``"umap"`` for example. It is the key ``"X_<basis>"`` in ``adata.obsm`` or, if
            there is no such key, ``basis`` itself.
        color: The column of ``adata.obs`` to color the cells by, or ``None`` for one color. A categorical (or text or
            boolean) column gives each category a color, and a numeric column a continuous color scale (``Oranges``).
            Cells with a missing value are light gray. A boolean column (or a categorical with boolean categories)
            is red for True and light gray for False, and the True cells are drawn on top of the others.
        hover_data: The columns of ``adata.obs`` to show in the tooltip, a name or a list of names.
        components: The two dimensions of the embedding to plot, as indices (starting at 0).
        palette: The colors of the categories, a list of colors (given to the categories in order, repeating if there
            are fewer colors than categories) or a dictionary from category to color. By default the colors in
            ``adata.uns["<color>_colors"]``, which scanpy stores, if there are any, and otherwise a long list of
            distinguishable colors that repeats if there are more categories than colors. For a boolean ``color`` the
            default is red for True and light gray for False; a list is then in the order ``[False, True]``.
        size: The diameter of the markers, in pixels. By default it decreases with the number of cells.
        title: The title of the figure.
        width: The width of the figure in pixels, by default (``None``) as wide as the space it is displayed in.
        height: The height of the figure in pixels, ``None`` for the default of plotly.
        show: Whether to also display the figure (in a notebook the figure that is returned is displayed anyway).
        save_html: The path of an HTML file to save the figure to, which must end in ``.html``. The file is
            self-contained (plotly.js is embedded in it, which adds about 5 MB), so it opens offline and can be shared.
            It is large for many cells, because the text of the tooltip of each cell is stored in it. An existing file
            is overwritten. The directory of the file must exist.

    Returns:
        The plotly figure.

    Raises:
        ImportError: If plotly is not installed.
        KeyError: If the embedding or a column is not found.
        ValueError: If ``save_html`` does not end in ``.html``.
        FileNotFoundError: If the directory of ``save_html`` does not exist.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "cellarium.ml.api.plotting.embedding needs plotly, which is not a dependency of cellarium-ml: "
            "install it with `pip install plotly`."
        ) from error

    if save_html is not None:  # checked first, so that a bad path does not cost the time of making the figure
        save_path = Path(save_html)
        if save_path.suffix.lower() != ".html":
            raise ValueError(f"save_html must be a path that ends in '.html', got '{save_html}'.")
        if not save_path.resolve().parent.is_dir():
            raise FileNotFoundError(f"The directory of save_html does not exist: '{save_path.resolve().parent}'.")

    key = _find_basis(adata, basis)
    coordinates = adata.obsm[key]
    coordinates = coordinates.to_numpy() if isinstance(coordinates, pd.DataFrame) else np.asarray(coordinates)
    if coordinates.ndim != 2 or len(components) != 2 or max(components) >= coordinates.shape[1] or min(components) < 0:
        raise ValueError(
            f"components {components} do not select two dimensions of obsm['{key}'], which has shape "
            f"{coordinates.shape}."
        )
    x, y = coordinates[:, components[0]], coordinates[:, components[1]]
    n_obs = len(x)

    hover_columns = [hover_data] if isinstance(hover_data, str) else list(hover_data or [])
    hover_columns = [column for column in hover_columns if column != color]  # the color is the title of the tooltip
    _check_obs_columns(adata, hover_columns, "hover_data")
    if color is not None:
        _check_obs_columns(adata, [color], "color")

    # the tooltip: the first line is the (bold) value of the color, then one line for each column of hover_data
    texts: list[np.ndarray] = []
    lines: list[str] = []
    categorical = False
    values: pd.Series | None = None
    if color is not None:
        values = adata.obs[color]
        categorical = not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
        if categorical:
            texts.append(_format_values(values))
        else:  # a number: its name is part of the title
            texts.append(np.array([f"{html.escape(str(color))}: {t}" for t in _format_values(values)], dtype=object))
        lines.append(f"<b>%{{customdata[{len(texts) - 1}]}}</b>")
    for column in hover_columns:
        texts.append(_format_values(adata.obs[column]))
        lines.append(f"{html.escape(str(column))}: %{{customdata[{len(texts) - 1}]}}")
    has_tooltip = len(texts) > 0
    customdata = np.stack(texts, axis=1) if has_tooltip else None
    hover = {
        "hovertemplate": "<br>".join(lines) + "<extra></extra>" if has_tooltip else None,
        "hoverinfo": None if has_tooltip else "skip",
    }

    marker_size = size if size is not None else float(np.clip(8 * (1000 / max(n_obs, 1)) ** 0.25, 2, 8))
    figure = go.Figure()

    def add_trace(selection: np.ndarray | slice, marker: dict[str, Any], **extra: Any) -> None:
        figure.add_trace(
            go.Scattergl(
                x=x[selection],
                y=y[selection],
                mode="markers",
                marker={"size": marker_size, **marker},
                customdata=None if customdata is None else customdata[selection],
                showlegend=False,
                **hover,
                **extra,
            )
        )

    everything = slice(None)
    if values is None:
        add_trace(everything, {"color": _NO_COLOR})
    elif categorical:
        categories = values if isinstance(values.dtype, pd.CategoricalDtype) else values.astype("category")
        is_bool = pd.api.types.is_bool_dtype(values)  # also a categorical whose categories are booleans
        if is_bool and palette is None:
            palette = {False: _FALSE_COLOR, True: _TRUE_COLOR}  # type: ignore[dict-item]
        colors = _categorical_colors(adata, color, categories, palette)  # type: ignore[arg-type]
        if is_bool:
            # drawn missing first, then False, then True, so that the (usually rare) True cells are on top
            codes = categories.cat.codes.to_numpy()
            levels = list(categories.cat.categories)
            is_missing = codes < 0
            is_true = (codes == levels.index(True)) if True in levels else np.zeros(n_obs, dtype=bool)
            for selection in (is_missing, ~is_missing & ~is_true, is_true):
                if selection.any():
                    # a tooltip with the color of the cell as its background
                    add_trace(
                        selection, {"color": list(colors[selection])}, hoverlabel={"bgcolor": list(colors[selection])}
                    )
        else:
            # a tooltip with the color of the cell as its background
            add_trace(everything, {"color": list(colors)}, hoverlabel={"bgcolor": list(colors)})
    else:
        numbers = values.to_numpy(dtype=float, na_value=np.nan)
        missing = np.isnan(numbers)
        if missing.any():  # drawn first, so below the others
            add_trace(missing, {"color": _NA_COLOR})
        from plotly.colors import sample_colorscale

        scale = sample_colorscale(_CONTINUOUS_SCALE, list(np.linspace(_CONTINUOUS_SCALE_START, 1, 11)))
        add_trace(
            ~missing,
            {
                "color": numbers[~missing],
                "colorscale": [[i / 10, c] for i, c in enumerate(scale)],
                "colorbar": {"title": {"text": str(color)}},
                "showscale": True,
            },
        )

    prefix = basis.removeprefix("X_").upper()
    axis = {"showgrid": False, "zeroline": False, "showticklabels": False, "ticks": ""}
    figure.update_layout(
        title=title,
        template="plotly_white",
        width=width,
        height=height,
        showlegend=False,
        xaxis={**axis, "title": f"{prefix}{components[0] + 1}"},
        yaxis={**axis, "title": f"{prefix}{components[1] + 1}", "scaleanchor": "x", "scaleratio": 1},
    )
    if save_html is not None:
        figure.write_html(save_path, include_plotlyjs=True, full_html=True)
    if show:
        figure.show()
    return figure
