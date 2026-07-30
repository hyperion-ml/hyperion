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
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
]

autosummary_generate = True
autosectionlabel_prefix_document = True

_intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    # Pin the inventory because PyTorch's stable URL redirects to a versioned
    # site and breaks fragment validation in Sphinx linkcheck.
    "torch": ("https://docs.pytorch.org/docs/2.8", None),
    "jsonargparse": ("https://jsonargparse.readthedocs.io/en/stable", None),
}

# A local documentation build must be reproducible without network access.
# Enable external inventory fetching explicitly in CI or when validating links.
docs_online = os.environ.get("HYPERION_DOCS_ONLINE", "").lower() in {
    "1",
    "true",
    "yes",
}
intersphinx_mapping = _intersphinx_mapping if docs_online else {}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
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

# Public docstrings contain illustrative ``>>>`` sessions that need corpus
# fixtures or optional runtime dependencies. Run only examples deliberately
# marked with Sphinx's ``testcode``/``testoutput`` directives; otherwise the
# doctest builder mistakes explanatory API examples for hermetic tests.
doctest_test_doctest_blocks = ""

# Strict builds use ``-W`` in docs/build.sh. Keep unresolved Python references
# visible without enabling nitpicky mode until the public API inventory has
# been converted to curated reference pages.
nitpicky = False

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
    "langcodes",
    "onnxruntime",
    "regex",
    "transformers",
    "datasets",
    "einops",
    "lhotse",
    "kaldialign",
    "loralib",
    "more_itertools",
    "pycountry",
    "sentencepiece",
    "tqdm",
    "utmosv2",
    "vox_profile",
    "wandb",
]
