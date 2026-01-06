# Configuration file for the Sphinx documentation builder.
#
# Full documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

# Add package to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

from shreamy import __version__

project = "shreamy"
author = "Gabriel Pfaffman"
copyright = f"{datetime.datetime.now().year}, {author}"  # noqa: A001
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    # Sphinx core extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    # MyST for Markdown support
    "myst_parser",
    # Copy button for code blocks
    "sphinx_copybutton",
    # Design elements (cards, grids, tabs)
    "sphinx_design",
]

# MyST configuration
myst_enable_extensions = [
    "amsmath",           # LaTeX math
    "colon_fence",       # ::: directive syntax
    "deflist",           # Definition lists
    "dollarmath",        # $math$ syntax
    "fieldlist",         # Field lists
    "html_admonition",   # HTML admonitions
    "html_image",        # HTML images
    "linkify",           # Auto-link URLs
    "replacements",      # Typography replacements
    "smartquotes",       # Smart quotes
    "strikethrough",     # ~~strikethrough~~
    "substitution",      # Substitutions
    "tasklist",          # Task lists
]

myst_heading_anchors = 3  # Generate anchors for h1-h3

# Source file suffixes
source_suffix = [".rst", ".md"]

# The master toctree document
master_doc = "index"

# Patterns to ignore
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "**/README.md",
]

# Default role for single backticks
default_role = "py:obj"

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Include __init__ docstring in class documentation
autoclass_content = "both"

# Generate autosummary stubs
autosummary_generate = True

# -- Napoleon configuration (Google/NumPy docstrings) -----------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "galpy": ("https://docs.galpy.org/en/latest/", None),
}

# -- HTML output configuration -----------------------------------------------

html_theme = "furo"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",  # Blue
        "color-brand-content": "#2563eb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/sgpfaff/shreamy",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_title = f"shreamy {version}"
html_short_title = "shreamy"

# Static files
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Favicon and logo (uncomment when you have these files)
# html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"

# -- Copy button configuration -----------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Todo extension configuration --------------------------------------------

todo_include_todos = True

# -- Options for other outputs -----------------------------------------------

# LaTeX configuration for PDF output
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "11pt",
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, "shreamy.tex", "shreamy Documentation", author, "manual"),
]

# Man page output
man_pages = [(master_doc, "shreamy", "shreamy Documentation", [author], 1)]

# Texinfo output
texinfo_documents = [
    (
        master_doc,
        "shreamy",
        "shreamy Documentation",
        author,
        "shreamy",
        "N-body simulations of stellar streams and shells from minor mergers.",
        "Miscellaneous",
    ),
]
