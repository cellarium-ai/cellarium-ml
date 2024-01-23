# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import glob
import os
import shutil

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Cellarium ML"
copyright = "2023, Cellarium AI"
author = "Cellarium AI"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []

# Disable documentation inheritance

autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"

# do not execute cells
nbsphinx_execute = "never"

# -- Copy notebook files

if not os.path.exists("tutorials"):
    os.makedirs("tutorials")

for src_file in glob.glob("../../notebooks/*.ipynb"):
    shutil.copy(src_file, "tutorials/")

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
