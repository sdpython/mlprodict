"""
@file Functions to creates a benchmark based on :epkg:`asv`
for many regressors and classifiers.
"""
import os
import json
import textwrap
import hashlib
import warnings
try:
    from pyquickhelper.pycode.code_helper import remove_extra_spaces_and_pep8
except ImportError:
    remove_extra_spaces_and_pep8 = lambda code, *args, **kwargs: code

try:
    from ..onnxrt.validate.validate_helper import (
        get_opset_number_from_onnx, sklearn_operators
    )
    from ..onnxrt.validate.validate import (
        _retrieve_problems_extra, _get_problem_data, _merge_options
    )
except (ValueError, ImportError):
    from mlprodict.onnxrt.validate.validate_helper import (
        get_opset_number_from_onnx, sklearn_operators
    )
    from mlprodict.onnxrt.validate.validate import (
        _retrieve_problems_extra, _get_problem_data, _merge_options
    )
try:
    from ..testing.verify_code import verify_code
except (ValueError, ImportError):
    from mlprodict.testing.verify_code import verify_code

# exec function does not import models but potentially
# requires all specific models used to defines scenarios
try:
    from ..onnxrt.validate.validate_scenarios import *  # pylint: disable=W0614,W0401
except ValueError:
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
    "pythons": ["3.7"],
    "matrix": {
        "cython": [],
        "jinja2": [],
        "joblib": [],
        "lightgbm": [],
        "numpy": [],
        "onnx": [],
        "onnxruntime": [],
        "pandas": [],
        "Pillow": [],
        "pybind11": [],
        "scipy": [],
        "skl2onnx": ["git+https://github.com/xadupre/sklearn-onnx.git@jenkins"],
        "scikit-learn": ["git+https://github.com/scikit-learn/scikit-learn.git"],
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


def create_asv_benchmark(
        location, opset_min=11, opset_max=None,
        runtime=('scikit-learn', 'python'), models=None,
        skip_models=None, extended_list=True,
        dims=(1, 10, 100, 10000, 100000),
        n_features=(4, 20), dtype=None,
        verbose=0, fLOG=print, clean=True,
        conf_params=None, filter_exp=None,
        filter_scenario=None, flat=False,
        exc=False, build=None, execute=False):
    """
    Creates an :epkg:`asv` benchmark in a folder
    but does not run it.

    :param n_features: number of features to try
    :param dims: number of observations to try
    :param verbose: integer from 0 (None) to 2 (full verbose)
    :param opset_min: tries every conversion from this minimum opset
    :param opset_max: tries every conversion up to maximum opset
    :param runtime: runtime to check, *scikit-learn*, *python*,
        *onnxruntime1* to check :epkg:`onnxruntime`,
        *onnxruntime2* to check every ONNX node independently
        with onnxruntime, many runtime can be checked at the same time
        if the value is a comma separated list
    :param models: list of models to test or empty
        string to test them all
    :param skip_models: models to skip
    :param extended_list: extends the list of :epkg:`scikit-learn` converters
        with converters implemented in this module
    :param n_features: change the default number of features for
        a specific problem, it can also be a comma separated list
    :param dtype: '32' or '64' or None for both,
        limits the test to one specific number types
    :param fLOG: logging function
    :param clean: clean the folder first, otherwise overwrites the content
    :param conf_params: to overwrite some of the configuration parameters
    :param filter_exp: function which tells if the experiment must be run,
        None to run all, takes *model, problem* as an input
    :param filter_scenario: second function which tells if the experiment must be run,
        None to run all, takes *model, problem, scenario, extra*
        as an input
    :param flat: one folder for all files or subfolders
    :param exc: if False, raises warnings instead of exceptions
        whenever possible
    :param build: where to put the outputs
    :param execute: execute each script to make sure
        imports are correct
    :return: created files

    The default configuration is the following:

    .. runpython::
        :showcode:

        import pprint
        from mlprodict.asv_benchmark.create_asv import default_asv_conf

        pprint.pprint(default_asv_conf)
    """
    # creates the folder if it does not exist.
    if not os.path.exists(location):
        if verbose > 0 and fLOG is not None:
            fLOG("[create_asv_benchmark] create folder '{}'.".format(location))
        os.makedirs(location)

    location_test = os.path.join(location, 'benches')
    if not os.path.exists(location_test):
        if verbose > 0 and fLOG is not None:
            fLOG("[create_asv_benchmark] create folder '{}'.".format(location_test))
        os.mkdir(location_test)

    # Cleans the content of the folder
    created = []
    if clean:
        for name in os.listdir(location_test):
            full_name = os.path.join(location_test, name)
            if os.path.isfile(full_name):
                os.remove(full_name)

    # configuration
    conf = default_asv_conf.copy()
    if conf_params is not None:
        for k, v in conf_params.items():
            conf[k] = v
    if build is not None:
        for fi in ['env_dir', 'results_dir', 'html_dir']:
            conf[fi] = os.path.join(build, conf[fi])
    dest = os.path.join(location, "asv.conf.json")
    created.append(dest)
    with open(dest, "w", encoding='utf-8') as f:
        json.dump(conf, f, indent=4)
    if verbose > 0 and fLOG is not None:
        fLOG("[create_asv_benchmark] create 'asv.conf.json'.")

    # __init__.py
    dest = os.path.join(location, "__init__.py")
    with open(dest, "w", encoding='utf-8') as f:
        pass
    created.append(dest)
    if verbose > 0 and fLOG is not None:
        fLOG("[create_asv_benchmark] create '__init__.py'.")
    dest = os.path.join(location_test, '__init__.py')
    with open(dest, "w", encoding='utf-8') as f:
        pass
    created.append(dest)
    if verbose > 0 and fLOG is not None:
        fLOG("[create_asv_benchmark] create 'benches/__init__.py'.")

    # flask_server
    tool_dir = os.path.join(location, 'tools')
    if not os.path.exists(tool_dir):
        os.mkdir(tool_dir)
    fl = os.path.join(tool_dir, 'flask_serve.py')
    with open(fl, "w", encoding='utf-8') as f:
        f.write(flask_helper)
    if verbose > 0 and fLOG is not None:
        fLOG("[create_asv_benchmark] create 'flask_serve.py'.")

    if verbose > 0 and fLOG is not None:
        fLOG("[create_asv_benchmark] create all tests.")

    created.extend(list(_enumerate_asv_benchmark_all_models(
        location_test, opset_min=opset_min, opset_max=opset_max,
        runtime=runtime, models=models,
        skip_models=skip_models, extended_list=extended_list,
        n_features=n_features, dtype=dtype,
        verbose=verbose, filter_exp=filter_exp,
        filter_scenario=filter_scenario,
        dims=dims, exc=exc, flat=flat,
        fLOG=fLOG, execute=execute)))

    if verbose > 0 and fLOG is not None:
        fLOG("[create_asv_benchmark] done.")
    return created


def _sklearn_subfolder(model):
    """
    Returns the list of subfolders for a model.
    """
    mod = model.__module__
    spl = mod.split('.')
    pos = spl.index('sklearn')
    res = spl[pos + 1: -1]
    if len(res) == 0:
        if spl[-1] == 'sklearn':
            res = ['_externals']
        elif spl[0] == 'sklearn':
            res = spl[pos + 1:]
        else:
            raise ValueError(
                "Unable to guess subfolder for '{}'.".format(model.__class__))
    res.append(model.__name__)
    return res


def _handle_init_files(model, flat, location, verbose, fLOG):
    "Returns created, location_model, prefix_import."
    if flat:
        return [], location, "."
    else:
        created = []
        subf = _sklearn_subfolder(model)
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
        return created, location_model, prefix_import


def _enumerate_asv_benchmark_all_models(  # pylint: disable=R0914
        location, opset_min=10, opset_max=None,
        runtime=('scikit-learn', 'python'), models=None,
        skip_models=None, extended_list=True,
        n_features=None, dtype=None,
        verbose=0, filter_exp=None,
        dims=None, filter_scenario=None,
        exc=True, flat=False, execute=False,
        fLOG=print):
    """
    Loops over all possible models and fills a folder
    with benchmarks following :epkg:`asv` concepts.

    :param n_features: number of features to try
    :param dims: number of observations to try
    :param verbose: integer from 0 (None) to 2 (full verbose)
    :param opset_min: tries every conversion from this minimum opset
    :param opset_max: tries every conversion up to maximum opset
    :param runtime: runtime to check, *scikit-learn*, *python*,
        *onnxruntime1* to check :epkg:`onnxruntime`,
        *onnxruntime2* to check every ONNX node independently
        with onnxruntime, many runtime can be checked at the same time
        if the value is a comma separated list
    :param models: list of models to test or empty
        string to test them all
    :param skip_models: models to skip
    :param extended_list: extends the list of :epkg:`scikit-learn` converters
        with converters implemented in this module
    :param n_features: change the default number of features for
        a specific problem, it can also be a comma separated list
    :param dtype: '32' or '64' or None for both,
        limits the test to one specific number types
    :param fLOG: logging function
    :param filter_exp: function which tells if the experiment must be run,
        None to run all, takes *model, problem* as an input
    :param filter_scenario: second function which tells if the experiment must be run,
        None to run all, takes *model, problem, scenario, extra*
        as an input
    :param exc: if False, raises warnings instead of exceptions
        whenever possible
    :param flat: one folder for all files or subfolders
    :param execute: execute each script to make sure
        imports are correct
    """

    ops = [_ for _ in sklearn_operators(extended=extended_list)]

    if models is not None:
        if not all(map(lambda m: isinstance(m, str), models)):
            raise ValueError("models must be a set of strings.")
        ops_ = [_ for _ in ops if _['name'] in models]
        if len(ops) == 0:
            raise ValueError("Parameter models is wrong: {}\n{}".format(
                models, ops[0]))
        ops = ops_
    if skip_models is not None:
        ops = [m for m in ops if m['name'] not in skip_models]

    if verbose > 0:

        def iterate():
            for i, row in enumerate(ops):
                fLOG("{}/{} - {}".format(i + 1, len(ops), row))
                yield row

        if verbose >= 11:
            verbose -= 10
            loop = iterate()
        else:
            try:
                from tqdm import trange

                def iterate_tqdm():
                    with trange(len(ops)) as t:
                        for i in t:
                            row = ops[i]
                            disp = row['name'] + " " * (28 - len(row['name']))
                            t.set_description("%s" % disp)
                            yield row

                loop = iterate_tqdm()

            except ImportError:
                loop = iterate()
    else:
        loop = ops

    if opset_max is None:
        opset_max = get_opset_number_from_onnx()
    opsets = list(range(opset_min, opset_max + 1))
    all_created = set()

    # loop on all models
    for row in loop:

        model = row['cl']

        problems, extras = _retrieve_problems_extra(
            model, verbose, fLOG, extended_list)
        if extras is None or problems is None:
            # Not tested yet.
            continue

        # flat or not flat
        created, location_model, prefix_import = _handle_init_files(
            model, flat, location, verbose, fLOG)
        for init in created:
            yield init

        # loops on problems
        for prob in problems:
            if filter_exp is not None and not filter_exp(model, prob):
                continue

            (X_train, X_test, y_train,
             y_test, Xort_test,
             init_types, conv_options, method_name,
             output_index, dofit, predict_kwargs) = _get_problem_data(prob, None)

            for scenario_extra in extras:
                subset_problems = None
                optimisations = None
                new_conv_options = None
                if len(scenario_extra) > 2:
                    options = scenario_extra[2]
                    if isinstance(options, dict):
                        subset_problems = options.get('subset_problems', None)
                        optimisations = options.get('optim', None)
                        new_conv_options = options.get('conv_options', None)
                    else:
                        subset_problems = options
                if subset_problems:
                    subset_problems = scenario_extra[2]
                    if prob not in subset_problems:
                        # Skips unrelated problem for a specific configuration.
                        continue
                scenario, extra = scenario_extra[:2]
                if optimisations is None:
                    optimisations = [None]
                if new_conv_options is None:
                    new_conv_options = [{}]

                if (filter_scenario is not None and
                        not filter_scenario(model, prob, scenario,
                                            extra, new_conv_options)):
                    continue

                if verbose >= 3 and fLOG is not None:
                    fLOG("[create_asv_benchmark] model={} scenario={} optim={} extra={} dofit={} (problem={})".format(
                        model.__name__, scenario, optimisations, extra, dofit, prob))
                created = _create_asv_benchmark_file(
                    location_model, opsets=opsets,
                    model=model, scenario=scenario, optimisations=optimisations,
                    extra=extra, dofit=dofit, problem=prob,
                    runtime=runtime, new_conv_options=new_conv_options,
                    X_train=X_train, X_test=X_test, y_train=y_train,
                    y_test=y_test, Xort_test=Xort_test,
                    init_types=init_types, conv_options=conv_options,
                    method_name=method_name, dims=dims, n_features=n_features,
                    output_index=output_index, predict_kwargs=predict_kwargs,
                    exc=exc, prefix_import=prefix_import,
                    execute=execute)
                for cr in created:
                    if cr in all_created:
                        raise RuntimeError(
                            "File '{}' was already created.".format(cr))
                    all_created.add(cr)
                    if verbose > 1 and fLOG is not None:
                        fLOG("[create_asv_benchmark] add '{}'.".format(cr))
                    yield cr


def _asv_class_name(model, scenario, optimisation,
                    extra, dofit, conv_options, problem):

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
        return r

    def clean_str_list(val):
        if val is None:
            return ""
        if isinstance(val, list):
            return ".".join(clean_str_list(v) for v in val if v)
        return clean_str(val)

    els = ['bench', model.__name__, scenario, clean_str(problem)]
    if not dofit:
        els.append('nofit')
    if extra:
        els.append(clean_str(extra))
    if optimisation:
        els.append(clean_str_list(optimisation))
    if conv_options:
        els.append(clean_str_list(conv_options))
    res = ".".join(els).replace("-", "_")
    if len(res) > 70:
        m = hashlib.sha256()
        m.update(res.encode('utf-8'))
        sh = m.hexdigest()
        if len(sh) > 6:
            sh = sh[:6]
        res = res[:70] + sh
    return res


def _create_asv_benchmark_file(  # pylint: disable=R0914
        location, model, scenario, optimisations, new_conv_options,
        extra, dofit, problem, runtime, X_train, X_test, y_train,
        y_test, Xort_test, init_types, conv_options,
        method_name, n_features, dims, opsets,
        output_index, predict_kwargs, prefix_import,
        exc, execute=False):
    """
    Creates a benchmark file based in the information received
    through the argument. It uses template @see cl TemplateBenchmark.
    """
    # Reads the template
    patterns = {}
    for suffix in ['classifier', 'regressor', 'clustering',
                   'outlier', 'trainable_transform', 'transform',
                   'multi_classifier', 'transform_positive']:
        template_name = os.path.join(os.path.dirname(
            __file__), "template", "skl_model_%s.py" % suffix)
        if not os.path.exists(template_name):
            raise FileNotFoundError(
                "Template '{}' was not found.".format(template_name))
        with open(template_name, "r", encoding="utf-8") as f:
            content = f.read()
        initial_content = '"""'.join(content.split('"""')[2:])
        patterns[suffix] = initial_content

    def pattern_problem(prob):
        if '-reg' in prob:
            return patterns['regressor']
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
        raise ValueError(
            "Unable to guess the right pattern for '{}'.".format(prob))

    def format_conv_options(d_options, class_name):
        if d_options is None:
            return None
        res = {}
        for k, v in d_options.items():
            if isinstance(k, type):
                if "." + class_name + "'" in str(k):
                    res[class_name] = v
                    continue
                raise ValueError(
                    "Class '{}', unable to format options {}".format(
                        class_name, d_options))
            res[k] = v
        return res

    def _nick_name_options(model, opts):
        if opts is None:
            return opts
        cdist = {model: {'optim': 'cdist'}}
        if opts == cdist:
            return 'cdist'
        res = {}
        for k, v in opts.items():
            if hasattr(k, '__name__'):
                res["####" + k.__name__ + "####"] = v
            else:
                res[k] = v
        return res

    runtimes_abb = {
        'scikit-learn': 'skl',
        'onnxruntime1': 'ort',
        'onnxruntime2': 'ort2',
        'python': 'pyrt',
    }
    runtime = [runtimes_abb[k] for k in runtime]

    # Looping over configuration.
    names = []
    for optimisation in optimisations:
        merged_options = [_merge_options(nconv_options, conv_options)
                          for nconv_options in new_conv_options]

        nck_opts = [_nick_name_options(model, opts)
                    for opts in merged_options]
        try:
            name = _asv_class_name(
                model, scenario, optimisation, extra,
                dofit, conv_options, problem)
        except ValueError as e:
            if exc:
                raise e
            warnings.warn(str(e))
            continue
        filename = name.replace(".", "_") + ".py"
        try:
            class_content = pattern_problem(problem)
        except ValueError as e:
            if exc:
                raise e
            warnings.warn(str(e))
            continue
        class_name = name.replace(
            "bench.", "").replace(".", "_") + "_bench"

        # n_features, N, runtimes
        rep = {
            "['skl', 'pyrt', 'ort'],  # values for runtime": str(runtime),
            "[1, 10, 100, 1000, 10000, 100000],  # values for N": str(dims),
            "[4, 20],  # values for nf": str(n_features),
            "[11],  # values for opset": str(opsets),
            "['float', 'double'],  # values for dtype":
                "['float']" if '-64' not in problem else "['float', 'double']",
            "[None],  # values for optim": "%r" % nck_opts,
        }
        for k, v in rep.items():
            if k not in content:
                raise ValueError("Unable to find '{}' in '{}'\n{}.".format(
                    k, template_name, content))
            class_content = class_content.replace(k, v + ',')
        class_content = class_content.split(
            "def _create_model(self):")[0].strip("\n ")
        if "####" in class_content:
            class_content = class_content.replace(
                "'####", "").replace("####'", "")
        if "####" in class_content:
            raise RuntimeError(
                "Substring '####' should not be part of the script for '{}'\n{}".format(
                    model.__name__, class_content))

        # Model setup
        class_content, atts = add_model_import_init(
            class_content, model, optimisation,
            extra, merged_options)
        class_content = class_content.replace(
            "class TemplateBenchmark",
            "class {}".format(class_name))

        # dtype, dofit
        atts.append("par_scenario = %r" % scenario)
        atts.append("par_problem = %r" % problem)
        atts.append("par_optimisation = %r" % optimisation)
        if not dofit:
            atts.append("par_dofit = False")
        if merged_options is not None and len(merged_options) > 0:
            atts.append("par_convopts = %r" % format_conv_options(
                conv_options, model.__name__))
        if atts:
            class_content = class_content.replace(
                "# additional parameters",
                "\n    ".join(atts))
        if prefix_import != '.':
            class_content = class_content.replace(
                " from .", "from .{}".format(prefix_import))

        # Check compilation
        try:
            compile(class_content, filename, 'exec')
        except SyntaxError as e:
            raise SyntaxError("Unable to compile model '{}'\n{}".format(
                model.__name__, class_content)) from e

        # Verifies missing imports.
        to_import, _ = verify_code(class_content, exc=False)
        try:
            miss = find_missing_sklearn_imports(to_import)
        except ValueError as e:
            raise ValueError(
                "Unable to check import in script\n{}".format(
                    class_content)) from e
        class_content = class_content.replace(
            "#  __IMPORTS__", "\n".join(miss))
        verify_code(class_content, exc=True)
        class_content = class_content.replace(
            "par_extra = {", "par_extra = {\n")
        class_content = remove_extra_spaces_and_pep8(
            class_content, aggressive=True)

        # Check compilation again
        try:
            obj = compile(class_content, filename, 'exec')
        except SyntaxError as e:
            raise SyntaxError("Unable to compile model '{}'\n{}".format(
                model.__name__,
                _display_code_lines(class_content))) from e

        # executes to check import
        if execute:
            try:
                exec(obj, globals(), locals())  # pylint: disable=W0122
            except Exception as e:
                raise RuntimeError(
                    "Unable to process class '{}' ('{}') a script due to '{}'\n{}".format(
                        model.__name__, filename, str(e),
                        _display_code_lines(class_content))) from e

        # Saves
        fullname = os.path.join(location, filename)
        names.append(fullname)
        with open(fullname, "w", encoding='utf-8') as f:
            f.write(class_content)

    return names


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
            raise ValueError(
                "Unable to interpret optimisation {}.".format(optimisation))

    # look for import place
    lines = class_content.split('\n')
    keep = None
    for pos, line in enumerate(lines):
        if "# Import specific to this model." in line:
            keep = pos
            break
    if keep is None:
        raise RuntimeError(
            "Unable to locate where to insert import in\n{}\n".format(class_content))

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

    imp_inst = "from {} import {}".format(mod, model.__name__)
    add_imports.append(imp_inst)
    add_imports.append("#  __IMPORTS__")
    lines[keep + 1] = "\n".join(add_imports)
    content = "\n".join(lines)

    # _create_model
    content = content.split('def _create_model(self):')[0].strip(' \n')
    lines = [content, "", "    def _create_model(self):"]
    if extra is not None and len(extra) > 0:
        lines.append("        return {}(".format(model.__name__))
        lines.append(_format_dict(extra, 12))
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
    if piece in {'LinearSVC', 'LinearSVR', 'NuSVR', 'SVR', 'SVC', 'NuSVC'}:
        import sklearn.svm
        glo[piece] = getattr(sklearn.svm, piece)
        return "sklearn.svm"
    if piece in {'KMeans'}:
        import sklearn.cluster
        glo[piece] = getattr(sklearn.cluster, piece)
        return "sklearn.cluster"
    raise ValueError("Unable to find module to import for '{}'.".format(piece))
