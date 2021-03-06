"""
@file Functions to creates a benchmark based on :epkg:`asv`
for many regressors and classifiers.
"""
import os
import textwrap
import hashlib
try:
    from ..onnxrt.optim.sklearn_helper import set_n_jobs
except (ValueError, ImportError):  # pragma: no cover
    from mlprodict.onnxrt.optim.sklearn_helper import set_n_jobs

# exec function does not import models but potentially
# requires all specific models used to defines scenarios
try:
    from ..onnxrt.validate.validate_scenarios import *  # pylint: disable=W0614,W0401
except (ValueError, ImportError):  # pragma: no cover
    # Skips this step if used in a benchmark.
    pass


default_asv_conf = {
    "version": 1,
    "project": "mlprodict",
    "project_url": "http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html",
    "repo": "https://github.com/sdpython/mlprodict.git",
    "repo_subdir": "",
    "install_command": ["python -mpip install {wheel_file}"],
    "uninstall_command": ["return-code=any python -mpip uninstall -y {project}"],
    "build_command": [
        "python setup.py build",
        "PIP_NO_BUILD_ISOLATION=false python -mpip wheel --no-deps --no-index -w {build_cache_dir} {build_dir}"
    ],
    "branches": ["master"],
    "environment_type": "virtualenv",
    "install_timeout": 600,
    "show_commit_url": "https://github.com/sdpython/mlprodict/commit/",
    # "pythons": ["__PYVER__"],
    "matrix": {
        "cython": [],
        "jinja2": [],
        "joblib": [],
        "lightgbm": [],
        "mlinsights": [],
        "numpy": [],
        "onnx": ["http://localhost:8067/simple/"],
        "onnxruntime": ["http://localhost:8067/simple/"],
        "pandas": [],
        "Pillow": [],
        "pybind11": [],
        "pyquickhelper": [],
        "scipy": [],
        # "git+https://github.com/xadupre/onnxconverter-common.git@jenkins"],
        "onnxconverter-common": ["http://localhost:8067/simple/"],
        # "git+https://github.com/xadupre/sklearn-onnx.git@jenkins"],
        "skl2onnx": ["http://localhost:8067/simple/"],
        # "git+https://github.com/scikit-learn/scikit-learn.git"],
        "scikit-learn": ["http://localhost:8067/simple/"],
        "xgboost": [],
    },
    "benchmark_dir": "benches",
    "env_dir": "env",
    "results_dir": "results",
    "html_dir": "html",
}

flask_helper = """
'''
Local ASV files do no properly render in a browser,
it needs to be served through a server.
'''
import os.path
from flask import Flask, Response

app = Flask(__name__)
app.config.from_object(__name__)


def root_dir():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "html")


def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except IOError as exc:
        return str(exc)


@app.route('/', methods=['GET'])
def mainpage():
    content = get_file('index.html')
    return Response(content, mimetype="text/html")


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def get_resource(path):  # pragma: no cover
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    complete_path = os.path.join(root_dir(), path)
    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    content = get_file(complete_path)
    return Response(content, mimetype=mimetype)


if __name__ == '__main__':  # pragma: no cover
    app.run(  # ssl_context=('cert.pem', 'key.pem'),
        port=8877,
        # host="",
    )
"""

pyspy_template = """
import sys
sys.path.append(r"__PATH__")
from __PYFOLD__ import __CLASSNAME__
import time
from datetime import datetime


def start():
    cl = __CLASSNAME__()
    cl.setup_cache()
    return cl


def profile0(iter, cl, runtime, N, nf, opset, dtype, optim):
    begin = time.perf_counter()
    for i in range(0, 100):
        cl.time_predict(runtime, N, nf, opset, dtype, optim)
    duration = time.perf_counter() - begin
    iter = max(100, int(25 / duration * 100)) # 25 seconds
    return iter


def setup_profile0(iter, cl, runtime, N, nf, opset, dtype, optim):
    cl.setup(runtime, N, nf, opset, dtype, optim)
    return profile0(iter, cl, runtime, N, nf, opset, dtype, optim)


def profile(iter, cl, runtime, N, nf, opset, dtype, optim):
    for i in range(iter):
        cl.time_predict(runtime, N, nf, opset, dtype, optim)
    return iter


def setup_profile(iter, cl, runtime, N, nf, opset, dtype, optim):
    cl.setup(runtime, N, nf, opset, dtype, optim)
    return profile(iter, cl, runtime, N, nf, opset, dtype, optim)


cl = start()
iter = None
print(datetime.now(), "begin")
"""


def _sklearn_subfolder(model):
    """
    Returns the list of subfolders for a model.
    """
    mod = model.__module__
    if mod is not None and mod.startswith('mlinsights'):
        return ['mlinsights', model.__name__]  # pragma: no cover
    spl = mod.split('.')
    try:
        pos = spl.index('sklearn')
    except ValueError as e:  # pragma: no cover
        raise ValueError(
            "Unable to find 'sklearn' in '{}'.".format(mod)) from e
    res = spl[pos + 1: -1]
    if len(res) == 0:
        if spl[-1] == 'sklearn':
            res = ['_externals']
        elif spl[0] == 'sklearn':
            res = spl[pos + 1:]
        else:
            raise ValueError(  # pragma: no cover
                "Unable to guess subfolder for '{}'.".format(model.__class__))
    res.append(model.__name__)
    return res


def _handle_init_files(model, flat, location, verbose, location_pyspy, fLOG):
    "Returns created, location_model, prefix_import."
    if flat:
        return ([], location, ".",
                (None if location_pyspy is None else location_pyspy))

    created = []
    subf = _sklearn_subfolder(model)
    subf = [_ for _ in subf if _[0] != '_' or _ == '_externals']
    location_model = os.path.join(location, *subf)
    prefix_import = "." * (len(subf) + 1)
    if not os.path.exists(location_model):
        os.makedirs(location_model)
        for fold in [location_model, os.path.dirname(location_model),
                     os.path.dirname(os.path.dirname(location_model))]:
            init = os.path.join(fold, '__init__.py')
            if not os.path.exists(init):
                with open(init, 'w') as _:
                    pass
                created.append(init)
                if verbose > 1 and fLOG is not None:
                    fLOG("[create_asv_benchmark] create '{}'.".format(init))
    if location_pyspy is not None:
        location_pyspy_model = os.path.join(location_pyspy, *subf)
        if not os.path.exists(location_pyspy_model):
            os.makedirs(location_pyspy_model)
    else:
        location_pyspy_model = None

    return created, location_model, prefix_import, location_pyspy_model


def _asv_class_name(model, scenario, optimisation,
                    extra, dofit, conv_options, problem,
                    shorten=True):

    def clean_str(val):
        s = str(val)
        r = ""
        for c in s:
            if c in ",-\n":
                r += "_"
                continue
            if c in ": =.+()[]{}\"'<>~":
                continue
            r += c
        for k, v in {'n_estimators': 'nest',
                     'max_iter': 'mxit'}.items():
            r = r.replace(k, v)
        return r

    def clean_str_list(val):
        if val is None:
            return ""  # pragma: no cover
        if isinstance(val, list):
            return ".".join(  # pragma: no cover
                clean_str_list(v) for v in val if v)
        return clean_str(val)

    els = ['bench', model.__name__, scenario, clean_str(problem)]
    if not dofit:
        els.append('nofit')
    if extra:
        if 'random_state' in extra and extra['random_state'] == 42:
            extra2 = extra.copy()
            del extra2['random_state']
            if extra2:
                els.append(clean_str(extra2))
        else:
            els.append(clean_str(extra))
    if optimisation:
        els.append(clean_str_list(optimisation))
    if conv_options:
        els.append(clean_str_list(conv_options))
    res = ".".join(els).replace("-", "_")

    if shorten:
        rep = {
            'ConstantKernel': 'Cst',
            'DotProduct': 'Dot',
            'Exponentiation': 'Exp',
            'ExpSineSquared': 'ExpS2',
            'GaussianProcess': 'GaussProc',
            'GaussianMixture': 'GaussMixt',
            'HistGradientBoosting': 'HGB',
            'LinearRegression': 'LinReg',
            'LogisticRegression': 'LogReg',
            'MultiOutput': 'MultOut',
            'OrthogonalMatchingPursuit': 'OrthMatchPurs',
            'PairWiseKernel': 'PW',
            'Product': 'Prod',
            'RationalQuadratic': 'RQ',
            'WhiteKernel': 'WK',
            'length_scale': 'ls',
            'periodicity': 'pcy',
        }
        for k, v in rep.items():
            res = res.replace(k, v)

        rep = {
            'Classifier': 'Clas',
            'Regressor': 'Reg',
            'KNeighbors': 'KNN',
            'NearestNeighbors': 'kNN',
            'RadiusNeighbors': 'RadNN',
        }
        for k, v in rep.items():
            res = res.replace(k, v)

        if len(res) > 70:  # shorten filename
            m = hashlib.sha256()
            m.update(res.encode('utf-8'))
            sh = m.hexdigest()
            if len(sh) > 6:
                sh = sh[:6]
            res = res[:70] + sh
    return res


def _read_patterns():
    """
    Reads the testing pattern.
    """
    # Reads the template
    patterns = {}
    for suffix in ['classifier', 'classifier_raw_scores', 'regressor', 'clustering',
                   'outlier', 'trainable_transform', 'transform',
                   'multi_classifier', 'transform_positive']:
        template_name = os.path.join(os.path.dirname(
            __file__), "template", "skl_model_%s.py" % suffix)
        if not os.path.exists(template_name):
            raise FileNotFoundError(  # pragma: no cover
                "Template '{}' was not found.".format(template_name))
        with open(template_name, "r", encoding="utf-8") as f:
            content = f.read()
        initial_content = '"""'.join(content.split('"""')[2:])
        patterns[suffix] = initial_content
    return patterns


def _select_pattern_problem(prob, patterns):
    """
    Selects a benchmark type based on the problem kind.
    """
    if '-reg' in prob:
        return patterns['regressor']
    if '-cl' in prob and '-dec' in prob:
        return patterns['classifier_raw_scores']
    if '-cl' in prob:
        return patterns['classifier']
    if 'cluster' in prob:
        return patterns['clustering']
    if 'outlier' in prob:
        return patterns['outlier']
    if 'num+y-tr' in prob:
        return patterns['trainable_transform']
    if 'num-tr-pos' in prob:
        return patterns['transform_positive']
    if 'num-tr' in prob:
        return patterns['transform']
    if 'm-label' in prob:
        return patterns['multi_classifier']
    raise ValueError(  # pragma: no cover
        "Unable to guess the right pattern for '{}'.".format(prob))


def _display_code_lines(code):
    rows = ["%03d %s" % (i + 1, line)
            for i, line in enumerate(code.split("\n"))]
    return "\n".join(rows)


def _format_dict(opts, indent):
    """
    Formats a dictionary as code.
    """
    rows = []
    for k, v in sorted(opts.items()):
        rows.append('%s=%r' % (k, v))
    content = ', '.join(rows)
    st1 = "\n".join(textwrap.wrap(content))
    return textwrap.indent(st1, prefix=' ' * indent)


def _additional_imports(model_name):
    """
    Adds additional imports for experimental models.
    """
    if model_name == 'IterativeImputer':
        return ["from sklearn.experimental import enable_iterative_imputer  # pylint: disable=W0611"]
    if model_name in ('HistGradientBoostingClassifier', 'HistGradientBoostingClassifier'):
        return ["from sklearn.experimental import enable_hist_gradient_boosting  # pylint: disable=W0611"]
    return None


def add_model_import_init(
        class_content, model, optimisation=None,
        extra=None, conv_options=None):
    """
    Modifies a template such as @see cl TemplateBenchmarkClassifier
    with code associated to the model *model*.

    @param  class_content       template (as a string)
    @param  model               model class
    @param  optimisation        model optimisation
    @param  extra               addition parameter to the constructor
    @param  conv_options        options for the conversion to ONNX
    @returm                     modified template
    """
    add_imports = []
    add_methods = []
    add_params = ["par_modelname = '%s'" % model.__name__,
                  "par_extra = %r" % extra]

    # additional methods and imports
    if optimisation is not None:
        add_imports.append(
            'from mlprodict.onnxrt.optim import onnx_optimisations')
        if optimisation == 'onnx':
            add_methods.append(textwrap.dedent('''
                def _optimize_onnx(self, onx):
                    return onnx_optimisations(onx)'''))
            add_params.append('par_optimonnx = True')
        elif isinstance(optimisation, dict):
            add_methods.append(textwrap.dedent('''
                def _optimize_onnx(self, onx):
                    return onnx_optimisations(onx, self.par_optims)'''))
            add_params.append('par_optims = {}'.format(
                _format_dict(optimisation, indent=4)))
        else:
            raise ValueError(  # pragma: no cover
                "Unable to interpret optimisation {}.".format(optimisation))

    # look for import place
    lines = class_content.split('\n')
    keep = None
    for pos, line in enumerate(lines):
        if "# Import specific to this model." in line:
            keep = pos
            break
    if keep is None:
        raise RuntimeError(  # pragma: no cover
            "Unable to locate where to insert import in\n{}\n".format(
                class_content))

    # imports
    loc_class = model.__module__
    sub = loc_class.split('.')
    if 'sklearn' not in sub:
        mod = loc_class
    else:
        skl = sub.index('sklearn')
        if skl == 0:
            if sub[-1].startswith("_"):
                mod = '.'.join(sub[skl:-1])
            else:
                mod = '.'.join(sub[skl:])
        else:
            mod = '.'.join(sub[:-1])

    exp_imports = _additional_imports(model.__name__)
    if exp_imports:
        add_imports.extend(exp_imports)
    imp_inst = (
        "try:\n    from {0} import {1}\nexcept ImportError:\n    {1} = None"
        "".format(mod, model.__name__))
    add_imports.append(imp_inst)
    add_imports.append("#  __IMPORTS__")
    lines[keep + 1] = "\n".join(add_imports)
    content = "\n".join(lines)

    # _create_model
    content = content.split('def _create_model(self):')[0].strip(' \n')
    lines = [content, "", "    def _create_model(self):"]
    if extra is not None and len(extra) > 0:
        lines.append("        return {}(".format(model.__name__))
        lines.append(_format_dict(set_n_jobs(model, extra), 12))
        lines.append("        )")
    else:
        lines.append("        return {}()".format(model.__name__))
    lines.append("")

    # methods
    for meth in add_methods:
        lines.append(textwrap.indent(meth, '    '))
        lines.append('')

    # end
    return "\n".join(lines), add_params


def find_missing_sklearn_imports(pieces):
    """
    Finds in :epkg:`scikit-learn` the missing pieces.

    @param      pieces      list of names in scikit-learn
    @return                 list of corresponding imports
    """
    res = {}
    for piece in pieces:
        mod = find_sklearn_module(piece)
        if mod not in res:
            res[mod] = []
        res[mod].append(piece)

    lines = []
    for k, v in res.items():
        lines.append("from {} import {}".format(
            k, ", ".join(sorted(v))))
    return lines


def find_sklearn_module(piece):
    """
    Finds the corresponding modulee for an element of :epkg:`scikit-learn`.

    @param      piece       name to import
    @return                 module name

    The implementation is not intelligence and should
    be improved. It is a kind of white list.
    """
    glo = globals()
    if piece in {'LinearRegression', 'LogisticRegression',
                 'SGDClassifier'}:
        import sklearn.linear_model
        glo[piece] = getattr(sklearn.linear_model, piece)
        return "sklearn.linear_model"
    if piece in {'DecisionTreeRegressor', 'DecisionTreeClassifier'}:
        import sklearn.tree
        glo[piece] = getattr(sklearn.tree, piece)
        return "sklearn.tree"
    if piece in {'ExpSineSquared', 'DotProduct', 'RationalQuadratic', 'RBF'}:
        import sklearn.gaussian_process.kernels
        glo[piece] = getattr(sklearn.gaussian_process.kernels, piece)
        return "sklearn.gaussian_process.kernels"
    if piece in {'LinearSVC', 'LinearSVR', 'NuSVR', 'SVR', 'SVC', 'NuSVC'}:  # pragma: no cover
        import sklearn.svm
        glo[piece] = getattr(sklearn.svm, piece)
        return "sklearn.svm"
    if piece in {'KMeans'}:  # pragma: no cover
        import sklearn.cluster
        glo[piece] = getattr(sklearn.cluster, piece)
        return "sklearn.cluster"
    if piece in {'OneVsRestClassifier', 'OneVsOneClassifier'}:  # pragma: no cover
        import sklearn.multiclass
        glo[piece] = getattr(sklearn.multiclass, piece)
        return "sklearn.multiclass"
    raise ValueError(  # pragma: no cover
        "Unable to find module to import for '{}'.".format(piece))
