# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import typing as t

from google.cloud import storage
from smart_open import open


def get_paths(paths: t.Union[str, t.List[str]]) -> t.List[str]:
    """
    Get a list of paths from a single path or a list of paths.

    :param paths: A single path or a list of paths.

    :return: A list of paths.
    """
    if isinstance(paths, list):
        return paths

    with open(paths, "r") as f:
        return f.read().splitlines()


def list_files_in_bucket(bucket_name, prefix=""):
    """
    List all files in a given GCS bucket directory.

    :param bucket_name: Name of the GCS bucket
    :param prefix: Directory path within the bucket (optional)
    :return: List of file names
    """
    # Initialize a client
    client = storage.Client()

    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    file_names = [blob.name for blob in blobs]

    return file_names
