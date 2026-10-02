# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Builds the C core (``deltacells._core``), which links against libzstd.

libzstd (headers and library) is looked up in this order: ``$ZSTD_ROOT``, ``$CONDA_PREFIX``, ``sys.prefix``,
Homebrew, ``/usr/local`` and ``/usr``. Set ``DELTACELLS_STATIC_ZSTD=1`` to link ``libzstd.a`` statically (useful for
wheels); otherwise the shared library is linked with an rpath.
"""

import os
import sys

from setuptools import Extension, setup


def find_zstd() -> dict:
    roots = [
        os.environ.get("ZSTD_ROOT"),
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        "/opt/homebrew",
        "/usr/local",
        "/usr",
    ]
    for root in filter(None, roots):
        include = os.path.join(root, "include")
        if not os.path.exists(os.path.join(include, "zstd.h")):
            continue
        for libdir in (
            os.path.join(root, "lib"),
            os.path.join(root, "lib64"),
            os.path.join(root, "lib", "x86_64-linux-gnu"),
        ):
            static = os.path.join(libdir, "libzstd.a")
            if os.environ.get("DELTACELLS_STATIC_ZSTD") == "1" and os.path.exists(static):
                return {"include_dirs": [include], "extra_objects": [static]}
            if any(
                f.startswith("libzstd.") and not f.endswith(".a") for f in os.listdir(libdir) if os.path.isdir(libdir)
            ):
                return {
                    "include_dirs": [include],
                    "library_dirs": [libdir],
                    "libraries": ["zstd"],
                    "extra_link_args": [f"-Wl,-rpath,{libdir}"],
                }
    raise RuntimeError(
        "libzstd (zstd.h and the library) was not found. Install it (e.g. `conda install zstd`, `brew install zstd`, "
        "`apt install libzstd-dev`) or point ZSTD_ROOT at its install prefix."
    )


ext = Extension(
    "deltacells._core",
    sources=["src/deltacells/_core.c"],
    extra_compile_args=["-O3", "-std=c11"],
    **find_zstd(),
)

setup(ext_modules=[ext])
