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

import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'SimVP'
copyright = '2022-2023, CAIRI'
author = 'CAIRI Authors'

# # The full version, including alpha/beta/rc tags
# version_file = '../../simvp/version.py'


# def get_version():
#     with open(version_file, 'r') as f:
#         exec(compile(f.read(), version_file, 'exec'))
#     return locals()['__version__']


# release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode',
    'sphinx_markdown_tables', 'sphinx_copybutton', 'myst_parser'
]

autodoc_mock_imports = ['json_tricks', 'simvp.version']

# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    # The target url that the logo directs to. Unset to do nothing
    'logo_url': 'https://simvpv2.readthedocs.io/en/latest/',
    # "menu" is a list of dictionaries where you can specify the content and the 
    # behavior of each item in the menu. Each item can either be a link or a
    # dropdown menu containing a list of links.
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/chengtan9907/SimVPv2'
        },
        {
            'name':
            'Upstream',
            'children': [
                {
                    'name': 'SimVP: Simpler yet Better Video Prediction',
                    'url': 'https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction',
                    'description': "CVPR'2022 paper SimVP: Simpler Yet Better Video Prediction"
                },
                {
                    'name': 'OpenMixup',
                    'url': 'https://github.com/Westlake-AI/openmixup',
                    'description': "CAIRI Supervised, Semi- and Self-Supervised Visual Representation Learning Toolbox and Benchmark"
                },
            ]
        },
    ],
    # For shared menu: If your project is a part of OpenMMLab's project and 
    # you would like to append Docs and OpenMMLab section to the right
    # of the menu, you can specify menu_lang to choose the language of
    # shared contents. Available options are 'en' and 'cn'. Any other
    # strings will fall back to 'en'.
    'menu_lang': 'en'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

language = 'en'

html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

# Enable ::: for my_st
myst_enable_extensions = ['colon_fence']
myst_heading_anchors = 4

master_doc = 'index'
