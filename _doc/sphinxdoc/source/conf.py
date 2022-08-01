# -*- coding: utf-8 -*-
import sys
import os
import pydata_sphinx_theme
from pyquickhelper.helpgen.default_conf import set_sphinx_variables
try:
    from mlprodict.onnx_conv import register_converters, register_rewritten_operators
except ImportError as e:
    raise ImportError(
        "Unable to import mlprodict. paths:\n{}".format(
            "\n".join(sys.path))) from e
register_converters()
try:
    register_rewritten_operators()
except KeyError:
    import warnings
    warnings.warn("converter for HistGradientBoosting* not not exist. "
                  "Upgrade sklearn-onnx.")

try:
    import generate_visual_graphs
    import generate_automated_pages
    import generate_onnx_ops
except ImportError:  # pragma: no cover
    this = os.path.dirname(__file__)
    sys.path.append(os.path.join(this, '_exts'))
    import generate_visual_graphs
    import generate_automated_pages
    import generate_onnx_ops


sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

set_sphinx_variables(
    __file__, "mlprodict", "Xavier Dupré", 2022,
    "pydata_sphinx_theme", "_static",
    locals(), extlinks=dict(
        issue=('https://github.com/sdpython/mlprodict/issues/%s', 'issue')),
    title="Python Runtime for ONNX", book=True)

blog_root = "http://www.xavierdupre.fr/app/mlprodict/helpsphinx/"
extensions.extend([
    'sphinxcontrib.blockdiag',
    'generate_automated_pages',
    'generate_visual_graphs',
    'generate_onnx_ops',
    'mlprodict.npy.xop_sphinx',
])

html_css_files = ['my-styles.css']
html_logo = "phdoc_static/project_ico.png"
html_sidebars = {}
language = "en"
mathdef_link_only = True
onnx_doc_folder = os.path.join(os.path.dirname(__file__), 'onnxops')

intersphinx_mapping.update({
    'cpyquickhelper': (
        'http://www.xavierdupre.fr/app/cpyquickhelper/helpsphinx/', None),
    'jyquickhelper': (
        'http://www.xavierdupre.fr/app/jyquickhelper/helpsphinx/', None),
    'lightgbm': ('https://lightgbm.readthedocs.io/en/latest/', None),
    'mlinsights': (
        'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/', None),
    'onnxmltools': (
        'http://www.xavierdupre.fr/app/onnxmltools/helpsphinx/', None),
    'onnxruntime': (
        'http://www.xavierdupre.fr/app/onnxruntime/helpsphinx/', None),
    'skl2onnx': ('http://onnx.ai/sklearn-onnx/', None),
    'torch': ('https://pytorch.org/docs/master/', None),
})

epkg_dictionary.update({
    '_PredictScorer': 'https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/scorer.py#L168',
    'airspeed-velocity': 'https://github.com/airspeed-velocity/asv',
    'ast': 'https://docs.python.org/3/library/ast.html',
    'asv': 'https://github.com/airspeed-velocity/asv',
    'bench1': 'http://www.xavierdupre.fr/app/mlprodict_bench/helpsphinx/index.html',
    'bench2': 'http://www.xavierdupre.fr/app/mlprodict_bench2/helpsphinx/index.html',
    'BLAS': 'http://www.netlib.org/blas/',
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
    'einsum': 'https://numpy.org/doc/stable/reference/generated/numpy.einsum.html',
    'exec': 'https://docs.python.org/3/library/functions.html#exec',
    'FunctionTransformer': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html',
    'GaussianProcessRegressor': 'https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html',
    'infer_shapes': 'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/onnx_python/shape_inference.html',
    'Iris': 'https://en.wikipedia.org/wiki/Iris_flower_data_set',
    'IR_VERSION': 'https://github.com/onnx/onnx/blob/master/docs/IR.md#onnx-versioning',
    'json': 'https://docs.python.org/3/library/json.html',
    'JSON': 'https://en.wikipedia.org/wiki/JSON',
    'joblib': 'https://joblib.readthedocs.io/en/latest/',
    'LAPACK': 'http://www.netlib.org/lapack/',
    'lightgbm': 'https://lightgbm.readthedocs.io/en/latest/',
    'make_attribute': 'https://github.com/onnx/onnx/blob/master/onnx/helper.py#L353',
    'make_scorer': 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html',
    'Minkowski distance': 'https://en.wikipedia.org/wiki/Minkowski_distance',
    'mlinsights': 'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/index.html',
    'mlprodict': 'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'mlstatpy': 'http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/index.html',
    'netron': 'https://github.com/lutzroeder/netron',
    'numba': 'https://numba.org/',
    'numpy': ('https://www.numpy.org/',
              ('https://docs.scipy.org/doc/numpy/reference/generated/numpy.{0}.html', 1),
              ('https://docs.scipy.org/doc/numpy/reference/generated/numpy.{0}.{1}.html', 2)),
    'openmp': 'https://www.openmp.org/',
    'ONNX': 'https://onnx.ai/',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX Backend': 'https://github.com/onnx/onnx/blob/main/docs/ImplementingAnOnnxBackend.md',
    'ONNX mappings': 'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/onnx_python/spec.html?highlight=mapping#type-mappings',
    'ONNX ML Operators': 'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md',
    'ONNX Operators': 'https://github.com/onnx/onnx/blob/master/docs/Operators.md',
    'ONNX Version': 'https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions',
    'ONNX Zoo': 'https://github.com/onnx/models',
    'onnxconverter_common': 'https://github.com/onnx/onnxmltools/tree/master/onnxutils/onnxconverter_common',
    'OnnxOperator': 'https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/algebra/onnx_operator.py#L116',
    'OnnxOperatorMixin': 'https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/algebra/onnx_operator_mixin.py#L16',
    'onnxruntime': 'https://github.com/microsoft/onnxruntime',
    'onnxruntime-extensions': 'https://github.com/microsoft/onnxruntime-extensions',
    'onnxruntime_perf_test': 'https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest',
    'Op': ('https://github.com/onnx/onnx/blob/master/docs/Operators.md',
           ('https://github.com/onnx/onnx/blob/master/docs/Operators.md#{0}', 1)),
    'opt-einsum': 'https://pypi.org/project/opt-einsum/',
    'pickle': 'https://docs.python.org/3/library/pickle.html',
    'pybind11': 'https://github.com/pybind/pybind11',
    'pypiserver': 'https://github.com/pypiserver/pypiserver',
    'pyspy': 'https://github.com/benfred/py-spy',
    'pytorch': 'https://pytorch.org/',
    'py-spy': 'https://github.com/benfred/py-spy',
    'Python': 'https://www.python.org/',
    'run_asv.bat': 'https://github.com/sdpython/mlprodict/blob/master/bin/run_asv.bat',
    'run_asv.sh': 'https://github.com/sdpython/mlprodict/blob/master/bin/run_asv.sh',
    'Rust': 'https://www.rust-lang.org/',
    'scikit-learn': 'https://scikit-learn.org/stable/',
    'skl2onnx': 'https://github.com/onnx/sklearn-onnx',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
    'sklearn-onnx tutorial': 'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/tutorial.html',
    'tensorflow': 'https://www.tensorflow.org/',
    'tensorflow-onnx': 'https://github.com/onnx/tensorflow-onnx',
    'tf2onnx': 'https://github.com/onnx/tensorflow-onnx',
    'Tokenizer': 'https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.Tokenizer',
    'torch': 'https://pytorch.org/',
    'tqdm': 'https://github.com/tqdm/tqdm',
    'TransferTransformer': 'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/mlinsights/mlmodel/transfer_transformer.html',
    'xgboost': "https://xgboost.readthedocs.io/en/latest/",
})

epkg_dictionary.update({
    'C_OrtDevice':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/'
        'onnxruntime_python/helpers.html#c-class-ortdevice',
    'C_OrtValue':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/'
        'onnxmd/onnxruntime_python/ortvalue.html#c-class-ortvaluevector',
    'C_SparseTensor':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#sparsetensor',
    'Contrib Operators':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_docs/ContribOperators.html',
    'FunctionProto':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/onnx_python/classes.html#functionproto',
    'Gemm':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-gemm-a-a-name-gemm-gemm-a',
    'If':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-if-a-a-name-if-if-a',
    'InferenceSession':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/inference.html'
        '#python-wrapper-for-inferencesession',
    'IOBinding':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#iobinding',
    'IR':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/IR.html',
    'Loop':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-loop-a-a-name-loop-loop-a',
    'ModelProto':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/onnx_python/classes.html#modelproto',
    'OnnxPipeline':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/'
        'sklapi/onnx_pipeline.html',
    'OneHotEncoder':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators-ml.html?highlight=onehotencoding'
        '#a-name-ai-onnx-ml-onehotencoder-a-a-name-ai-onnx-'
        'ml-onehotencoder-ai-onnx-ml-onehotencoder-a',
    'ORTModule':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/'
        'api/onnxruntime_python/training_torch.html#ortmodule',
    'OrtModuleGraphBuilder':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html'
        "#ortmodulegraphbuilder",
    'OrtModuleGraphBuilderConfiguration':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html'
        "#ortmodulegraphbuilderconfiguration",
    'OrtDevice':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#ortdevice',
    'OrtValue':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#ortvalue',
    'OrtValueCache':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html#ortvaluecache',
    'OrtValueVector':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/training_session.html#ortvaluevector',
    'PartialGraphExecutionState':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html'
        "#partialgraphexecutionstate",
    'RunOptions':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/inference.html#runoptions',
    'Scan':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-scan-a-a-name-scan-scan-a',
    'SessionIOBinding':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/'
        'onnxruntime_python/inference.html#'
        'onnxruntime.capi._pybind_state.SessionIOBinding',
    'SessionOptions':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/inference.html#sessionoptions',
    'SparseTensor':
        'http://www.xavierdupre.fr/app/onnxruntime_training/'
        'helpsphinx/api/tensors.html#sparsetensor',
    'TfIdfVectorizer':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#'
        'a-name-tfidfvectorizer-a-a-name-tfidfvectorizer-tfidfvectorizer-a',
    'TopK':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html#a-name-topk-a-a-name-topk-topk-a',
    'TrainingAgent':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnxruntime_python/training_partial.html'
        "#trainingagent",
    'TrainingParameters':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/'
        'onnxruntime_python/training.html#trainingparameters',
    'TrainingSession':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/'
        'onnxruntime_python/training.html#onnxruntime.TrainingSession',
    'Transpose':
        'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxmd/'
        'onnx_docs/Operators.html'
        '#a-name-transpose-a-a-name-transpose-transpose-a',
    'TreeEnsembleRegressor':
        'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md'
        '#ai.onnx.ml.TreeEnsembleRegressor',
})
