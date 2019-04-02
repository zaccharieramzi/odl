# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function
import sphinx
import glob
import os
import sphinx_rtd_theme
import sys
from packaging.version import parse as parse_version

# --- General configuration --- #

# All configuration values have a default; values that are commented out
# serve to show the default.

try:
    # Verify that we can import odl
    import odl
except Exception as e:
    print('Failed importing odl, exiting', file=sys.stderr)
    print(e, file=sys.stderr)
    sys.exit(1)


# Add numpydoc folder
sys.path.insert(0, os.path.abspath('../numpydoc'))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'numpydoc'
]
# Use newer 'imgmath' extension if possible
if parse_version(sphinx.__version__) >= parse_version('1.4'):
    extensions.append('sphinx.ext.imgmath')
else:
    extensions.append('sphinx.ext.pngmath')

# Add external links to GitHub
extlinks = {
    'pull': ('https://github.com/odlgroup/odl/pull/%s', 'PR '),
    'issue': ('https://github.com/odlgroup/odl/issues/%s', 'issue '),
    'commit': ('https://github.com/odlgroup/odl/commit/%s', 'commit ')
}


# Intersphinx to get Numpy and other targets
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pywt': ('https://pywavelets.readthedocs.io/en/latest/', None),
    'pyfftw': ('https://pyfftw.readthedocs.io/en/latest/', None),
    'pynfft': ('https://pythonhosted.org/pyNFFT/', None),
    'pytest': ('https://docs.pytest.org/en/latest/', None)}


# Stop autodoc from skipping __init__
def skip(app, what, name, obj, skip, options):
    if (name.startswith('__') and name.endswith('__') and
        name not in ['__abstractmethods__',
                     '__doc__',
                     '__hash__',
                     '__module__',
                     '__dict__',
                     '__weakref__']):
        return False
    if name in ['_multiply',
                '_divide',
                '_lincomb',
                '_call']:
        return False
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# Autosummary
autosummary_generate = glob.glob("./*.rst")

# Stops WARNING: toctree contains reference to nonexisting document
# (not anymore since Sphinx 1.6)
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = True

# Set order to mirror source
autodoc_member_order = 'bysource'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'odl'
copyright = u'2015-2017, ODL development group'
author = u'Jonas Adler, Holger Kohr, Ozan Öktem'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = odl.__version__
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['*.py', '*.pyc']

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'any'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# Warn on dead links and other stuff
nitpicky = True
nitpick_ignore = [('py:class', 'future.types.newobject.newobject')]

# --- Options for HTML output --- #

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = 'ODL'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'odldoc'

# --- Options for LaTeX output --- #

latex_elements = {
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{enumitem}

\setlistdepth{9}
'''
}
# The paper size ('letterpaper' or 'a4paper').
# 'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
# 'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
# 'preamble': '',

# Latex figure (float) alignment
# 'figure_align': 'htbp',


# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'odl.tex', u'ODL Documentation',
     u'Jonas Adler, Holger Kohr, Ozan Öktem', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# --- Options for manual page output --- #

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'odl', u'ODL Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
# man_show_urls = False


# --- Options for Texinfo output --- #

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'odl', u'ODL Documentation',
     author, 'odl', 'One line description of project.',
     'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False
