"""Sphinx configuration for prismo documentation."""

import sys
from importlib.metadata import PackageNotFoundError, version as get_version
from pathlib import Path
from typing import Any

_basedir = Path(__file__).parent
sys.path.insert(0, str(_basedir / ".." / "app"))

project = "prismo"
try:
    release = get_version("prismo")
except PackageNotFoundError:
    release = "0.0.0"
copyright = "Copyright &copy; 2026, The Prismo Community"
author = "The Prismo Community"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
    "sphinx_iconify",
    "sphinx_autodoc_typehints",
    "sphinx_sitemap",
]

html_theme = "shibuya"
html_title = "prismo"
html_logo = "_static/prismo-name.svg"
html_favicon = "_static/prismo.svg"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_baseurl = "https://benvial.github.io/prismo/"
html_copy_source = False
html_show_sourcelink = False

html_extra_path: list[str] = []

html_theme_options = {
    "accent_color": "cyan",
    "discussion_url": "https://github.com/benvial/prismo/discussions",
    "github_url": "https://github.com/benvial/prismo",
    "globaltoc_expand_depth": 1,
    "light_logo": "_static/prismo-name.svg",
    "dark_logo": "_static/prismo-name-dark.svg",
}

html_context = {
    "source_type": "github",
    "source_user": "benvial",
    "source_repo": "prismo",
}

templates_path = ["_templates"]

# sphinx-autodoc-typehints
autodoc_typehints = "description"
typehints_document_rtype = False
typehints_use_rtype = False
always_use_bars_union = True
typehints_fully_qualified = False
typehints_defaults = "comma"
always_document_param_types = True
autodoc_typehints_description_target = "documented"
typehints_use_signature = True
typehints_use_signature_return = False

# Napoleon
napoleon_use_rtype = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__init__, __new__, __init_subclass__",
}

language = "en"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "research",
    "agents",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}


def setup(app: Any) -> None:
    pass
