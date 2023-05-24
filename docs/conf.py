# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information

project = 'AgileRL'
copyright = '2023, AgileRL'
author = 'Nick Ustaran-Anderegg'

release = '0.1'
version = '0.1.6'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "https://user-images.githubusercontent.com/47857277/223126514-8b3131a2-1fde-4a6e-bb93-e12eb45af785.png"
html_theme_options = {
    'logo_only': False,
    'display_version': False,
    'navigation_depth': 3,
    "collapse_navigation": False
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
