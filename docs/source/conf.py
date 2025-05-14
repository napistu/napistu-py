# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Napistu Python library'
copyright = '2025, Sean Hackett'
author = 'Sean Hackett'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # Auto-generate API docs
    'sphinx.ext.napoleon',  # Support Google/NumPy docstrings
    'myst_parser',          # Markdown support
    'sphinx.ext.autosummary'
]

# Autogenerate stub files
autosummary_generate = True

# Include more members
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# separate main and utility functions

def custom_autodoc_process_docstring(app, what, name, obj, options, lines):
    """
    Custom processing to separate main and utility functions
    """
    # Logic to add separators or custom formatting
    if name.startswith('_'):
        # Add custom formatting for utility functions
        lines.insert(0, '**Utility Function**')
        lines.insert(1, '')

def setup(app):
    app.connect('autodoc-process-docstring', custom_autodoc_process_docstring)