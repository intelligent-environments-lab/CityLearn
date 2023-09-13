# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from citylearn.__init__ import __version__


# -- Project information -----------------------------------------------------

project = 'CityLearn'
copyright = '2023, Jose Ramon Vazquez-Canteli, Kingsley Nweye, Zoltan Nagy'
author = 'Jose Ramon Vazquez-Canteli, Kingsley Nweye, Zoltan Nagy'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinxcontrib.bibtex', # for citations
    'sphinxemoji.sphinxemoji', # for emojis
    'sphinx_copybutton', # to copy code block
    'myst_nb', # jupyter notebook
    'nbsphinx_link', # link jupyter notebook from dir that is not in docs/
    'sphinx_panels', # for backgrounds
]

# source for bib references
bibtex_bibfiles = ['references.bib']

# citation style
bibtex_default_style = 'plain'
bibtex_reference_style = 'label'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

#html context
html_context = {
  'display_github': True,
  'github_user': 'intelligent-environments-lab',
  'github_repo': 'CityLearn',
  'github_version': 'master/docs/source/',
}