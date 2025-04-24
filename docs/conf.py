import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath(os.path.join('..', 'HODDIES')))
from _version import __version__

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'HODDIES'
copyright = '2025, Antoine Rocher'
author = 'Antoine Rocher'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
html_static_path = []
html_title = 'HODDIES'
html_theme_options = {
    'repository_url': 'https://github.com/abacusorg/abacusutils',
    'repository_branch': 'main',
    # "launch_buttons": {
    #     "binderhub_url": "https://mybinder.org",
    #     "notebook_interface": "jupyterlab",
    #     "colab_url": "https://colab.research.google.com/",
    # },
    'use_edit_page_button': True,
    'use_issues_button': True,
    'use_repository_button': True,
    'use_download_button': True,
    'use_fullscreen_button': False,
    'path_to_docs': 'docs/',
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

root_doc = 'index'
source_suffix = {
    '.rst': 'restructuredtext',
}

html_theme = 'sphinx_rtd_theme'
exclude_patterns = ['build', '**.ipynb_checkpoints']
autodoc_mock_imports = ['pycorr', 'mpytools']

# -- Options for EPUB output
epub_show_urls = 'footnote'

git_repo = 'https://github.com/cosmodesi/pycorr.git'
git_root = 'https://github.com/cosmodesi/pycorr/blob/main/'
extlinks = {'root': (git_root + '%s', '%s')}

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


