# -*- coding: utf-8 -*-
import sys
import os
import sphinx_readable_theme
from pyquickhelper.helpgen.default_conf import set_sphinx_variables, get_default_stylesheet
from sklearn.experimental import enable_hist_gradient_boosting
from mlprodict.onnx_conv import register_converters, register_rewritten_operators
register_converters()
try:
    register_rewritten_operators()
except KeyError:
    import warnings
    warnings.warn("converter for HistGradientBoosting* not not exist. "
                  "Upgrade sklearn-onnx")

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
    'sphinxcontrib.blockdiag',
    'generate_automated_pages',
    'generate_visual_graphs',
])

html_context = {
    'css_files': get_default_stylesheet() + ['_static/my-styles.css', '_static/gallery.css'],
}

html_logo = "phdoc_static/project_ico.png"

html_sidebars = {}

language = "en"

mathdef_link_only = True

epkg_dictionary.update({
    '_PredictScorer': 'https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/scorer.py#L168',
    'airspeed-velocity': 'https://github.com/airspeed-velocity/asv',
    'asv': 'https://github.com/airspeed-velocity/asv',
    'bench1': 'http://www.xavierdupre.fr/app/benches/mlprodict_bench/index.html',
    'bench2': 'http://www.xavierdupre.fr/app/benches/mlprodict_bench2/index.html',
    'C': "https://en.wikipedia.org/wiki/C_(programming_language)",
    'cdist': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html',
    'cffi': "https://cffi.readthedocs.io/en/latest/",
    'Converters with options': 'http://www.xavierdupre.fr/app/sklearn-onnx/helpsphinx/parameterized.html',
    'coo_matrix': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html',
    'csv': 'https://en.wikipedia.org/wiki/Comma-separated_values',
    'cython': 'https://cython.org/',
    "DataFrame": "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html",
    'dot': 'https://en.wikipedia.org/wiki/DOT_(graph_description_language)',
    'DOT': 'https://en.wikipedia.org/wiki/DOT_(graph_description_language)',
    'exec': 'https://docs.python.org/3/library/functions.html#exec',
    'Iris': 'https://en.wikipedia.org/wiki/Iris_flower_data_set',
    'IR_VERSION': 'https://github.com/onnx/onnx/blob/master/docs/IR.md#onnx-versioning',
    'json': 'https://docs.python.org/3/library/json.html',
    'JSON': 'https://en.wikipedia.org/wiki/JSON',
    'joblib': 'https://joblib.readthedocs.io/en/latest/',
    'lightgbm': 'https://lightgbm.readthedocs.io/en/latest/',
    'make_scorer': 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html',
    'Minkowski distance': 'https://en.wikipedia.org/wiki/Minkowski_distance',
    'mlinsights': '',
    'mlprodict': 'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'openmp': 'https://www.openmp.org/',
    'ONNX': 'https://onnx.ai/',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX Operators': 'https://github.com/onnx/onnx/blob/master/docs/Operators.md',
    'ONNX ML Operators': 'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md',
    'onnxconverter_common': 'https://github.com/onnx/onnxmltools/tree/master/onnxutils/onnxconverter_common',
    'OnnxOperatorMixin': 'https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/algebra/onnx_operator_mixin.py#L16',
    'onnxruntime': 'https://github.com/microsoft/onnxruntime',
    'onnxruntime_perf_test': 'https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest',
    'pybind11': 'https://github.com/pybind/pybind11',
    'pypiserver': 'https://github.com/pypiserver/pypiserver',
    'pyspy': 'https://github.com/benfred/py-spy',
    'py-spy': 'https://github.com/benfred/py-spy',
    'Python': 'https://www.python.org/',
    'run_asv.bat': 'https://github.com/sdpython/mlprodict/blob/master/bin/run_asv.bat',
    'run_asv.sh': 'https://github.com/sdpython/mlprodict/blob/master/bin/run_asv.sh',
    'Rust': 'https://www.rust-lang.org/',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
    'Tokenizer': 'https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.Tokenizer',
    'TransferTransformer': 'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/mlinsights/mlmodel/transfer_transformer.html',
    'xgboost': "https://xgboost.readthedocs.io/en/latest/",
})
