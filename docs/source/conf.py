# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Cellarium ML"
copyright = "2023, Cellarium AI"
author = "Cellarium AI"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
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
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
