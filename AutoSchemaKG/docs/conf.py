# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AutoSchemaKG'
copyright = '2025, Hong Ting TSANG (Dennis)'
author = 'Hong Ting TSANG (Dennis)'
release = 'v0.0.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Auto-generate API docs from code
    'sphinx.ext.napoleon',     # Support Google/NumPy docstrings
    'sphinx.ext.viewcode',     # Link to source code
    'sphinx.ext.intersphinx',  # Cross-reference external docs
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os
import sys
html_static_path = ['_static']
html_logo = '_static/icon.png'  # Add your logo to docs/_static/
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,  # Deep hierarchy like ASER
}
