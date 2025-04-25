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
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath(os.path.join('..', 'HODDIES')))
from _version import __version__

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme'
]

# -- Project information -----------------------------------------------------

project = 'HODDIES'
copyright = '2025, Antoine Rocher'

# The full version, including alpha/beta/rc tags
release = __version__

html_theme = 'sphinx_rtd_theme'

autodoc_mock_imports = ['pycorr', 'colossus', 'cosmoprimo']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', '**.ipynb_checkpoints']

root_doc = 'index'
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

git_repo = 'https://github.com/antoine-rocher/HODDIES.git'
git_root = 'https://github.com/antoine-rocher/HODDIES/blob/main/'

extlinks = {'root': (git_root + '%s', '%s')}

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None)
}

# thanks to: https://github.com/sphinx-doc/sphinx/issues/4054#issuecomment-329097229
def _replace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result


ultimate_replacements = {
    '{gitrepo}': git_repo
}

def setup(app):
    app.add_config_value('ultimate_replacements', {}, True)
    app.connect('source-read',_replace)


autoclass_content = 'both'



# import os
# import sys
# sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath(os.path.join('..', 'HODDIES')))
# from _version import __version__

# # Configuration file for the Sphinx documentation builder.

# # -- Project information

# project = 'HODDIES'
# copyright = '2025, Antoine Rocher'
# author = 'Antoine Rocher'

# release = '0.1'
# version = '0.1.0'

# # -- General configuration

# extensions = [
#     'sphinx.ext.duration',
#     'sphinx.ext.doctest',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.intersphinx',
#     'sphinx.ext.napoleon',
#     'sphinx.ext.autosectionlabel',
# ]

# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
# }
# intersphinx_disabled_domains = ['std']

# templates_path = ['_templates']

# # -- Options for HTML output
# html_static_path = []
# html_title = 'HODDIES'
# html_theme_options = {
#     'repository_url': 'https://github.com/abacusorg/abacusutils',
#     'repository_branch': 'main',
#     # "launch_buttons": {
#     #     "binderhub_url": "https://mybinder.org",
#     #     "notebook_interface": "jupyterlab",
#     #     "colab_url": "https://colab.research.google.com/",
#     # },
#     'use_edit_page_button': True,
#     'use_issues_button': True,
#     'use_repository_button': True,
#     'use_download_button': True,
#     'use_fullscreen_button': False,
#     'path_to_docs': 'docs/',
# }

# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = True

# root_doc = 'index'
# source_suffix = {
#     '.rst': 'restructuredtext',
# }

# html_theme = 'sphinx_rtd_theme'
# exclude_patterns = ['build', '**.ipynb_checkpoints']
# autodoc_mock_imports = ['pycorr', 'mpytools']

# # -- Options for EPUB output
# epub_show_urls = 'footnote'

# git_repo = 'https://github.com/cosmodesi/pycorr.git'
# git_root = 'https://github.com/cosmodesi/pycorr/blob/main/'
# extlinks = {'root': (git_root + '%s', '%s')}

# # thanks to: https://github.com/sphinx-doc/sphinx/issues/4054#issuecomment-329097229
# def _replace(app, docname, source):
#     result = source[0]
#     for key in app.config.ultimate_replacements:
#         result = result.replace(key, app.config.ultimate_replacements[key])
#     source[0] = result


# ultimate_replacements = {
#     '{gitrepo}': git_repo
# }

# def setup(app):
#     app.add_config_value('ultimate_replacements', {}, True)
#     app.connect('source-read',_replace)


# autoclass_content = 'both'
