# -*- coding: utf-8 -*-
import sys
import os
import sphinx_readable_theme
from pyquickhelper.helpgen.default_conf import set_sphinx_variables, get_default_stylesheet
try:
    import generate_visual_graphs
    import generate_automated_pages
except ImportError:
    this = os.path.dirname(__file__)
    sys.path.append(os.path.join(this, '_exts'))
    import generate_visual_graphs
    import generate_automated_pages


sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

set_sphinx_variables(__file__, "mlprodict", "Xavier Dupré", 2019,
                     "readable", sphinx_readable_theme.get_html_theme_path(),
                     locals(), extlinks=dict(
                         issue=('https://github.com/sdpython/mlprodict/issues/%s', 'issue')),
                     title="Python Runtime for ONNX", book=True)

blog_root = "http://www.xavierdupre.fr/app/mlprodict/helpsphinx/"
extensions.extend([
    'generate_automated_pages',
    'generate_visual_graphs',
])

html_context = {
    'css_files': get_default_stylesheet() + ['_static/my-styles.css', '_static/gallery.css'],
}

html_logo = "project_ico.png"

html_sidebars = {}

language = "en"

mathdef_link_only = True

epkg_dictionary.update({
    '_PredictScorer': 'https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/scorer.py#L168',
    'C': "https://en.wikipedia.org/wiki/C_(programming_language)",
    'cffi': "https://cffi.readthedocs.io/en/latest/",
    "DataFrame": "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html",
    'cdist': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html',
    'dot': 'https://en.wikipedia.org/wiki/DOT_(graph_description_language)',
    'DOT': 'https://en.wikipedia.org/wiki/DOT_(graph_description_language)',
    'Iris': 'https://en.wikipedia.org/wiki/Iris_flower_data_set',
    'json': 'https://docs.python.org/3/library/json.html',
    'JSON': 'https://en.wikipedia.org/wiki/JSON',
    'lightgbm': 'https://lightgbm.readthedocs.io/en/latest/',
    'Minkowski distance': 'https://en.wikipedia.org/wiki/Minkowski_distance',
    'mlprodict': 'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'ONNX': 'https://onnx.ai/',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX Operators': 'https://github.com/onnx/onnx/blob/master/docs/Operators.md',
    'ONNX ML Operators': 'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md',
    'onnxconverter_common': 'https://github.com/onnx/onnxmltools/tree/master/onnxutils/onnxconverter_common',
    'OnnxOperatorMixin': 'https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/algebra/onnx_operator_mixin.py#L16',
    'onnxruntime': 'https://github.com/microsoft/onnxruntime',
    'Python': 'https://www.python.org/',
    'Rust': 'https://www.rust-lang.org/',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
    'xgboost': "https://xgboost.readthedocs.io/en/latest/",
})
