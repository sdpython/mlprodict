"""
@file Functions to creates a benchmark based on :epkg:`asv`
for many regressors and classifiers.
"""
import os
import json
from ..onnxrt.validate.validate_problems import _problems, find_suitable_problem
from ..onnxrt.validate.validate_scenarios import _extra_parameters
from ..onnxrt.validate.validate_helper import (
    get_opset_number_from_onnx, sklearn_operators
)
from ..onnxrt.validate.validate import _retrieve_problems_extra, _get_problem_data


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
    # The DVCS being used.  If not set, it will be automatically
    # determined from "repo" by looking at the protocol in the URL
    # (if remote), or by looking for special directories, such as
    # ".git" (if local).
    # "dvcs": "git",
    "environment_type": "virtualenv",
    "install_timeout": 600,
    "show_commit_url": "http://github.com/sdpython/mlprodict/commit/",
    "pythons": ["3.7"],
    "matrix": {
        "cython": [],
        "jinja2": [],
        "joblib": [],
        "numpy": [],
        "onnx": [],
        "onnxruntime": [],
        "pandas": [],
        "Pillow": [],
        "pybind11": [],
        "scipy": [],
        "skl2onnx": [],
        "scikit-learn": [],
    },
    "benchmark_dir": ".",
    "env_dir": "build/env",
    "results_dir": "build/results",
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
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "html")


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
        location, opset_min=9, opset_max=None,
        runtime=['scikit-learn', 'python'], models=None,
        skip_models=None, extended_list=True,
        dims=[1, 100, 10000], n_features=[4, 20], dtype=None,
        verbose=0, fLOG=print, clean=True,
        conf_params=None, filter_exp=None,
        filter_scenario=None):
    """
    Creates an :epkg:`asv` benchmark in a folder.

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

    conf = default_asv_conf.copy()
    if conf_params is not None:
        for k, v in conf_params.items():
            conf[k] = v
    dest = os.path.join(location, "asv.conf.json")
    created.append(dest)
    with open(dest, "w", encoding='utf-8') as f:
        json.dump(conf, f)
    if verbose > 0 and fLOG is not None:
        fLOG("[create_asv_benchmark] create 'asv.conf.json'.")

    fl = os.path.join(location, 'flask_serve.py')
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
        dims=dims, fLOG=fLOG)))

    if verbose > 0 and fLOG is not None:
        fLOG("[create_asv_benchmark] done.")
    return created


def _enumerate_asv_benchmark_all_models(
        location, opset_min=9, opset_max=None,
        runtime=['scikit-learn', 'python'], models=None,
        skip_models=None, extended_list=True,
        n_features=None, dtype=None,
        verbose=0, filter_exp=None,
        dims=None, filter_scenario=None, fLOG=print):
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
    created = []

    for row in loop:

        model = row['cl']

        problems, extras = _retrieve_problems_extra(
            model, verbose, fLOG, extended_list)
        if extras is None or problems is None:
            # Not tested yet.
            continue

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
                    location,
                    model=model, scenario=scenario, optimisations=optimisations,
                    extra=extra, dofit=dofit, prob=prob,
                    runtime=runtime, new_conv_options=new_conv_options,
                    X_train=X_train, X_test=X_test, y_train=y_train,
                    y_test=y_test, Xort_test=Xort_test,
                    init_types=init_types, conv_options=conv_options,
                    method_name=method_name, dims=dims, n_features=n_features,
                    output_index=output_index, predict_kwargs=predict_kwargs)
                for cr in created:
                    if verbose > 1 and fLOG is not None:
                        fLOG("[create_asv_benchmark] add '{}'.".format(cr))
                    yield cr


def _asv_class_name(model, prob, scenario, optimisation, extra, dofit, conv_options):

    def clean_str(val):
        s = str(val)
        s = s.replace("{", "")
        s = s.replace("}", "")
        s = s.replace(",", "")
        s = s.replace("\"", "")
        s = s.replace("'", "")
        s = s.replace(":", "_")
        s = s.replace(" ", "")
        s = s.replace("-", "_")
        return s

    p = prob.replace("~", "")
    els = ['bench', model.__name__, p, scenario]
    if not dofit:
        els.append('nofit')
    if extra:
        els.append(clean_str(extra))
    if optimisation:
        els.append(clean_str(optimisation))
    if conv_options:
        els.append(clean_str(conv_options))
    return ".".join(els).replace("-", "_")


def _create_asv_benchmark_file(
        location, model, scenario, optimisations, new_conv_options,
        extra, dofit, prob, runtime, X_train, X_test, y_train,
        y_test, Xort_test, init_types, conv_options,
        method_name, n_features, dims,
        output_index, predict_kwargs):
    """
    Creates a benchmark file based in the information received
    through the argument. It uses template @see cl TemplateBenchmark.
    """
    # Reads the template
    patterns = {}
    for suffix in ['classifier', 'regressor']:
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
        raise ValueError(
            "Unable to guess the right pattern for '{}'.".format(prob))

    runtimes_abb = {
        'scikit-learn': 'skl',
        'onnxruntime1': 'ort',
        'onnxruntime2': 'ort2',
        'python': 'pyrt',
    }
    runtime = [runtimes_abb[k] for k in runtime]

    # Looping over configuration.
    names = []
    for conv_options in new_conv_options:
        for optimisation in optimisations:
            name = _asv_class_name(
                model, prob, scenario, optimisation, extra,
                dofit, conv_options)
            filename = name + ".py"
            names.append(filename)
            class_content = pattern_problem(prob)

            # n_features, N, runtimes
            rep = {
                "['skl', 'pyrt', 'ort'],  # values for runtime": str(runtime),
                "[1, 100, 10000],  # values for N": str(dims),
                "[4, 20],  # values for nf": str(n_features),
            }
            for k, v in rep.items():
                if k not in content:
                    raise ValueError("Unable to find '{}' in '{}'\n{}.".format(
                        k, template_name, content))
                class_content = class_content.replace(k, v + ',')
            class_content = class_content.split(
                "def _create_model(self):")[0].strip("\n ")

            # Model setup

            # Model inference

            # Check compilation
            compile(class_content, filename, 'exec')

    return names
