# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

try:
    from hyperion import __version__ as hyperion_version
except Exception:
    hyperion_version = "0.0.0"

project = "Hyperion"
author = "Jesus Villalba"
copyright = "2020-2026, Johns Hopkins University"
version = hyperion_version
release = hyperion_version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
htmlhelp_basename = "hyperiondoc"

latex_documents = [
    (master_doc, "hyperion.tex", "Hyperion Documentation", author, "manual"),
]
man_pages = [(master_doc, "hyperion", "Hyperion Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "hyperion",
        "Hyperion Documentation",
        author,
        "hyperion",
        "Speech processing toolkit documentation.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_exclude_files = ["search.html"]

todo_include_todos = False

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Keep docs build lightweight on RTD by mocking heavy/optional dependencies.
autodoc_mock_imports = [
    "torch",
    "torchaudio",
    "torchvision",
    "soundfile",
    "sndfile",
    "librosa",
    "onnxruntime",
    "transformers",
    "datasets",
    "lhotse",
    "kaldialign",
    "sentencepiece",
    "utmosv2",
    "vox_profile",
    "wandb",
]
