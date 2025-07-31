import requests

import gcsfs
import google.cloud.storage as gcs
import h5py
import numpy as np


class SeekableHTTPFile:
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self._cache = {}
        
    def read(self, size=-1):
        if size == -1:
            # Read from current position to end
            response = requests.get(self.url, headers={'Range': f'bytes={self.pos}-'})
            data = response.content
            self.pos += len(data)
            return data
        else:
            # Read specific number of bytes
            end_pos = self.pos + size - 1
            cache_key = (self.pos, end_pos)
            
            if cache_key not in self._cache:
                response = requests.get(self.url, headers={'Range': f'bytes={self.pos}-{end_pos}'})
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
            size = int(response.headers.get('content-length', 0))
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


def get_h5ad_files_n_cells(h5ad_paths: list[str]) -> list[int]:
    """
    Get the number of cells in each h5ad file in a list of paths.
    """
    n_cells_list = []
    for h5ad_path in h5ad_paths:
        n_cells = _h5ad_file_read_elem(h5ad_path, fun=lambda x: x["obs/_index"].shape[0])
        n_cells_list.append(n_cells)

    return n_cells_list


def get_h5ad_file_var_names_g(h5ad_path: str) -> np.ndarray:
    """
    Get var_names_g from an h5ad file.
    """
    var_names_g = _h5ad_file_read_elem(h5ad_path, fun=lambda x: x["var/_index"][:])
    return var_names_g.astype(str)


def _h5ad_file_read_elem(h5ad_path: str, fun: callable) -> np.ndarray | float | int:
    """
    Read info from an h5ad file, loading as little of it as possible.
    """
    def _gcloud_version(h5ad_path: str) -> int:
        fs = gcsfs.GCSFileSystem()
        with fs.open(h5ad_path, "rb") as f:
            with h5py.File(f) as h5handle:
                return fun(h5handle)
    
    def _local_version(h5ad_path: str) -> int:
        with h5py.File(h5ad_path, "r") as h5handle:
            return fun(h5handle)
    
    def _url_version(h5ad_path: str) -> int:
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
    client = gcs.Client()
    bucket = gs_bucket_path.split("/")[2]
    prefix = "/".join(gs_bucket_path.split("/")[3:]) + "/"
    blobs = client.list_blobs(bucket)
    return [f"gs://{bucket}/{blob.name}" for blob in blobs if blob.name.startswith(prefix) and blob.name.endswith(".h5ad")]
