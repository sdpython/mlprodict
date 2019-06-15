# -*- coding: utf-8 -*-
import sys
import os
import sphinx_readable_theme
from pyquickhelper.helpgen.default_conf import set_sphinx_variables, get_default_stylesheet


sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

set_sphinx_variables(__file__, "mlprodict", "Xavier Dupré", 2019,
                     "readable", sphinx_readable_theme.get_html_theme_path(),
                     locals(), extlinks=dict(
                         issue=('https://github.com/sdpython/mlprodict/issues/%s', 'issue')),
                     title="mlprodict", book=True)

blog_root = "http://www.xavierdupre.fr/app/mlprodict/helpsphinx/"

html_context = {
    'css_files': get_default_stylesheet() + ['_static/my-styles.css'],
}

html_logo = "project_ico.png"

html_sidebars = {}

language = "en"

mathdef_link_only = True

epkg_dictionary.update({
    'C': "https://en.wikipedia.org/wiki/C_(programming_language)",
    'cffi': "https://cffi.readthedocs.io/en/latest/",
    'DOT': 'https://en.wikipedia.org/wiki/DOT_(graph_description_language)',
    'json': 'https://docs.python.org/3/library/json.html',
    'JSON': 'https://en.wikipedia.org/wiki/JSON',
    'ONNX': 'https://onnx.ai/',
    'onnx': 'https://github.com/onnx/onnx',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
    'onnxruntime': 'https://github.com/microsoft/onnxruntime',
    'Python': 'https://www.python.org/',
    'xgboost': "https://xgboost.readthedocs.io/en/latest/",
})
