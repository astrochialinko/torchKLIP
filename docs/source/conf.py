# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

project = 'torchklip'
copyright = '2025, Chia-Lin Ko'
author = 'Chia-Lin Ko'
root_doc = 'index'
release = 'v0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'nbsphinx',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx']

# Automatically generate stub pages for any autosummary directives
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/astrochialinko/torchKLIP",  # Replace with your repo
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "navbar_align": "content",
    "navbar_center": ["navbar-nav"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
}

html_css_files = [
    './css/custom.css',
]


def setup(app):
    app.add_js_file("https://d3js.org/d3.v7.min.js")  # load D3.js from CDN
    app.add_js_file("./js/tutorials_origindata.js")
    app.add_js_file("./js/tutorials_step1.js")
    # app.add_js_file("./js/tutorials_step2.js")
    app.add_js_file("./js/tutorials_pca_dataset.js")
    app.add_js_file("./js/tutorials_psf_dataset.js")
    app.add_js_file("./js/parms_klip.js")
    app.add_js_file("./js/parms_annular_vis.js")
