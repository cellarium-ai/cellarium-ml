# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import concurrent.futures
from typing import Callable

import gcsfs
import h5py
import numpy as np
import requests
from tqdm import tqdm


class SeekableHTTPFile:
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self._cache = {}

    def read(self, size=-1):
        if size == -1:
            # Read from current position to end
            response = requests.get(self.url, headers={"Range": f"bytes={self.pos}-"})
            data = response.content
            self.pos += len(data)
            return data
        else:
            # Read specific number of bytes
            end_pos = self.pos + size - 1
            cache_key = (self.pos, end_pos)

            if cache_key not in self._cache:
                response = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end_pos}"})
                self._cache[cache_key] = response.content

            data = self._cache[cache_key]
            self.pos += len(data)
            return data

    def seek(self, pos, whence=0):
        if whence == 0:  # SEEK_SET
            self.pos = pos
        elif whence == 1:  # SEEK_CUR
            self.pos += pos
        elif whence == 2:  # SEEK_END
            # Get file size first
            response = requests.head(self.url)
            size = int(response.headers.get("content-length", 0))
            self.pos = size + pos
        return self.pos

    def tell(self):
        return self.pos

    def close(self):
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _h5py_read_n_obs(h5handle: h5py.File) -> int:
    idx_col = h5handle["obs"].attrs["_index"]
    try:
        n_obs = h5handle[f"obs/{idx_col}"].shape[0]
    except AttributeError:
        # can happen if somehow the obs index is saved as a categorical (not supposed to be allowed)
        n_obs = h5handle[f"obs/{idx_col}/codes"].shape[0]
    return n_obs


def _h5py_read_var_names(h5handle: h5py.File) -> np.ndarray:
    idx_col = h5handle["var"].attrs["_index"]
    try:
        var_names = h5handle[f"var/{idx_col}"][:]
    except AttributeError:
        # can happen if somehow the var index is saved as a categorical (not supposed to be allowed)
        var_names = h5handle[f"var/{idx_col}/categories"][:]
    return var_names


def get_h5ad_file_n_cells(h5ad_path: str) -> int:
    """
    Get the number of cells in each h5ad file in a list of paths.
    """
    n_cells = _h5ad_file_read_elem(h5ad_path, fun=_h5py_read_n_obs)
    assert isinstance(n_cells, int), "Expected int from _h5py_read_n_obs"
    return n_cells


def get_h5ad_files_n_cells(h5ad_paths: list[str]) -> list[int]:
    """
    Get the number of cells in each h5ad file in a list of paths.
    ThreadPoolExecutor is used (preserves order).
    """
    # return [get_h5ad_file_n_cells(h5ad_path) for h5ad_path in h5ad_paths]
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        return list(
            tqdm(
                executor.map(get_h5ad_file_n_cells, h5ad_paths),
                total=len(h5ad_paths),
                desc="Reading n_obs from h5ad files",
                unit="file",
            )
        )


def get_h5ad_files_limits(h5ad_paths: list[str]) -> np.ndarray:
    """
    Return the `limits` to be used in constructing a :class:`~cellarium.ml.data.DistributedAnnDataCollection`
    based on sizes of the provided h5ad files.
    """
    limits = np.cumsum(get_h5ad_files_n_cells(h5ad_paths))
    return limits


def get_h5ad_file_var_names_g(h5ad_path: str) -> np.ndarray:
    """
    Get var_names_g from an h5ad file.
    """
    var_names_g = _h5ad_file_read_elem(h5ad_path, fun=_h5py_read_var_names)
    assert isinstance(var_names_g, np.ndarray), "Expected numpy array from _h5py_read_var_names"
    return var_names_g.astype(str)


def _h5ad_file_read_elem(h5ad_path: str, fun: Callable[[h5py.File], int | np.ndarray]) -> np.ndarray | int:
    """
    Read info from an h5ad file, loading as little of it as possible.
    """

    def _gcloud_version(h5ad_path: str) -> int | np.ndarray:
        fs = gcsfs.GCSFileSystem()
        with fs.open(h5ad_path, "rb") as f:
            with h5py.File(f) as h5handle:
                return fun(h5handle)

    def _local_version(h5ad_path: str) -> int | np.ndarray:
        with h5py.File(h5ad_path, "r") as h5handle:
            return fun(h5handle)

    def _url_version(h5ad_path: str) -> int | np.ndarray:
        """Optimized version that streams only the needed parts of the file"""
        with SeekableHTTPFile(h5ad_path) as f:
            with h5py.File(f, "r") as h5handle:
                return fun(h5handle)

    if h5ad_path.startswith("gs://"):
        out = _gcloud_version(h5ad_path)
    elif h5ad_path.startswith("http://") or h5ad_path.startswith("https://"):
        out = _url_version(h5ad_path)
    else:
        out = _local_version(h5ad_path)

    return out


def h5ad_paths_from_google_bucket(gs_bucket_path: str) -> list[str]:
    """
    Helper function to get h5ad file paths from a Google Cloud Storage bucket, like a Cellarium Nexus curriculum
    """
    if not gs_bucket_path.startswith("gs://"):
        raise ValueError("Invalid Google Cloud Storage bucket path -- must start with 'gs://'")
    fs = gcsfs.GCSFileSystem()
    paths = fs.ls(gs_bucket_path[5:])
    return [f"gs://{path}" for path in paths if path.endswith(".h5ad")]
