"""
@file Functions to creates a benchmark based on :epkg:`asv`
for many regressors and classifiers.
"""
import os
import sys
import json
import textwrap
import warnings
import re
from pyquickhelper.pycode.code_helper import remove_extra_spaces_and_pep8
try:
    from ._create_asv_helper import (
        default_asv_conf,
        flask_helper,
        pyspy_template,
        _handle_init_files,
        _asv_class_name,
        _read_patterns,
        _select_pattern_problem,
        _display_code_lines,
        add_model_import_init,
        find_missing_sklearn_imports)
except ImportError:  # pragma: no cover
    from mlprodict.asv_benchmark._create_asv_helper import (
        default_asv_conf,
        flask_helper,
        pyspy_template,
        _handle_init_files,
        _asv_class_name,
        _read_patterns,
        _select_pattern_problem,
        _display_code_lines,
        add_model_import_init,
        find_missing_sklearn_imports)

try:
    from ..tools.asv_options_helper import (
        get_opset_number_from_onnx, shorten_onnx_options)
    from ..onnxrt.validate.validate_helper import sklearn_operators
    from ..onnxrt.validate.validate import (
        _retrieve_problems_extra, _get_problem_data, _merge_options)
except (ValueError, ImportError):  # pragma: no cover
    from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx
    from mlprodict.onnxrt.validate.validate_helper import sklearn_operators
    from mlprodict.onnxrt.validate.validate import (
        _retrieve_problems_extra, _get_problem_data, _merge_options)
    from mlprodict.tools.asv_options_helper import shorten_onnx_options
try:
    from ..testing.verify_code import verify_code
except (ValueError, ImportError):  # pragma: no cover
    from mlprodict.testing.verify_code import verify_code

# exec function does not import models but potentially
# requires all specific models used to define scenarios
try:
    from ..onnxrt.validate.validate_scenarios import *  # pylint: disable=W0614,W0401
except (ValueError, ImportError):  # pragma: no cover
    # Skips this step if used in a benchmark.
    pass


def create_asv_benchmark(
        location, opset_min=-1, opset_max=None,
        runtime=('scikit-learn', 'python_compiled'), models=None,
        skip_models=None, extended_list=True,
        dims=(1, 10, 100, 10000),
        n_features=(4, 20), dtype=None,
        verbose=0, fLOG=print, clean=True,
        conf_params=None, filter_exp=None,
        filter_scenario=None, flat=False,
        exc=False, build=None, execute=False,
        add_pyspy=False, env=None,
        matrix=None):
    """
    Creates an :epkg:`asv` benchmark in a folder
    but does not run it.

    :param n_features: number of features to try
    :param dims: number of observations to try
    :param verbose: integer from 0 (None) to 2 (full verbose)
    :param opset_min: tries every conversion from this minimum opset,
        -1 to get the current opset defined by module :epkg:`onnx`
    :param opset_max: tries every conversion up to maximum opset,
        -1 to get the current opset defined by module :epkg:`onnx`
    :param runtime: runtime to check, *scikit-learn*, *python*,
        *python_compiled* compiles the graph structure
        and is more efficient when the number of observations is
        small, *onnxruntime1* to check :epkg:`onnxruntime`,
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
    :param add_pyspy: add an extra folder with code to profile
        each configuration
    :param env: None to use the default configuration or ``same`` to use
        the current one
    :param matrix: specifies versions for a module,
        example: ``{'onnxruntime': ['1.1.1', '1.1.2']}``,
        if a package name starts with `'~'`, the package is removed
    :return: created files

    The default configuration is the following:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import pprint
        from mlprodict.asv_benchmark.create_asv import default_asv_conf

        pprint.pprint(default_asv_conf)

    The benchmark does not seem to work well with setting
    ``-environment existing:same``. The publishing fails.
    """
    if opset_min == -1:
        opset_min = get_opset_number_from_onnx()
    if opset_max == -1:
        opset_max = get_opset_number_from_onnx()  # pragma: no cover
    if verbose > 0 and fLOG is not None:  # pragma: no cover
        fLOG("[create_asv_benchmark] opset in [{}, {}].".format(
            opset_min, opset_max))

    # creates the folder if it does not exist.
    if not os.path.exists(location):
        if verbose > 0 and fLOG is not None:  # pragma: no cover
            fLOG("[create_asv_benchmark] create folder '{}'.".format(location))
        os.makedirs(location)  # pragma: no cover

    location_test = os.path.join(location, 'benches')
    if not os.path.exists(location_test):
        if verbose > 0 and fLOG is not None:
            fLOG("[create_asv_benchmark] create folder '{}'.".format(location_test))
        os.mkdir(location_test)

    # Cleans the content of the folder
    created = []
    if clean:
        for name in os.listdir(location_test):
            full_name = os.path.join(location_test, name)  # pragma: no cover
            if os.path.isfile(full_name):  # pragma: no cover
                os.remove(full_name)

    # configuration
    conf = default_asv_conf.copy()
    if conf_params is not None:
        for k, v in conf_params.items():
            conf[k] = v
    if build is not None:
        for fi in ['env_dir', 'results_dir', 'html_dir']:  # pragma: no cover
            conf[fi] = os.path.join(build, conf[fi])
    if env == 'same':
        if matrix is not None:
            raise ValueError(  # pragma: no cover
                "Parameter matrix must be None if env is 'same'.")
        conf['pythons'] = ['same']
        conf['matrix'] = {}
    elif matrix is not None:
        drop_keys = set(p for p in matrix if p.startswith('~'))
        matrix = {k: v for k, v in matrix.items() if k not in drop_keys}
        conf['matrix'] = {k: v for k,
                          v in conf['matrix'].items() if k not in drop_keys}
        conf['matrix'].update(matrix)
    elif env is not None:
        raise ValueError(  # pragma: no cover
            "Unable to handle env='{}'.".format(env))
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

    # command line
    if sys.platform.startswith("win"):
        run_bash = os.path.join(tool_dir, 'run_asv.bat')  # pragma: no cover
    else:
        run_bash = os.path.join(tool_dir, 'run_asv.sh')
    with open(run_bash, 'w') as f:
        f.write(textwrap.dedent("""
            echo --BENCHRUN--
            python -m asv run --show-stderr --config ./asv.conf.json
            echo --PUBLISH--
            python -m asv publish --config ./asv.conf.json -o ./html
            echo --CSV--
            python -m mlprodict asv2csv -f ./results -o ./data_bench.csv
            """))

    # pyspy
    if add_pyspy:
        dest_pyspy = os.path.join(location, 'pyspy')
        if not os.path.exists(dest_pyspy):
            os.mkdir(dest_pyspy)
    else:
        dest_pyspy = None

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
        fLOG=fLOG, execute=execute,
        dest_pyspy=dest_pyspy)))

    if verbose > 0 and fLOG is not None:
        fLOG("[create_asv_benchmark] done.")
    return created


def _enumerate_asv_benchmark_all_models(  # pylint: disable=R0914
        location, opset_min=10, opset_max=None,
        runtime=('scikit-learn', 'python'), models=None,
        skip_models=None, extended_list=True,
        n_features=None, dtype=None,
        verbose=0, filter_exp=None,
        dims=None, filter_scenario=None,
        exc=True, flat=False, execute=False,
        dest_pyspy=None, fLOG=print):
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
    :param dest_pyspy: add a file to profile the prediction
        function with :epkg:`pyspy`
    """

    ops = [_ for _ in sklearn_operators(extended=extended_list)]
    patterns = _read_patterns()

    if models is not None:
        if not all(map(lambda m: isinstance(m, str), models)):
            raise ValueError(
                "models must be a set of strings.")  # pragma: no cover
        ops_ = [_ for _ in ops if _['name'] in models]
        if len(ops) == 0:
            raise ValueError("Parameter models is wrong: {}\n{}".format(  # pragma: no cover
                models, ops[0]))
        ops = ops_
    if skip_models is not None:
        ops = [m for m in ops if m['name'] not in skip_models]

    if verbose > 0:

        def iterate():
            for i, row in enumerate(ops):  # pragma: no cover
                fLOG("{}/{} - {}".format(i + 1, len(ops), row))
                yield row

        if verbose >= 11:
            verbose -= 10  # pragma: no cover
            loop = iterate()  # pragma: no cover
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

            except ImportError:  # pragma: no cover
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
            continue  # pragma: no cover

        # flat or not flat
        created, location_model, prefix_import, dest_pyspy_model = _handle_init_files(
            model, flat, location, verbose, dest_pyspy, fLOG)
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

                if subset_problems and isinstance(subset_problems, (list, set)):
                    if prob not in subset_problems:
                        # Skips unrelated problem for a specific configuration.
                        continue
                elif subset_problems is not None:
                    raise RuntimeError(  # pragma: no cover
                        "subset_problems must be a set or a list not {}.".format(
                            subset_problems))

                scenario, extra = scenario_extra[:2]
                if optimisations is None:
                    optimisations = [None]
                if new_conv_options is None:
                    new_conv_options = [{}]

                if (filter_scenario is not None and
                        not filter_scenario(model, prob, scenario,
                                            extra, new_conv_options)):
                    continue  # pragma: no cover

                if verbose >= 3 and fLOG is not None:
                    fLOG("[create_asv_benchmark] model={} scenario={} optim={} extra={} dofit={} (problem={} method_name='{}')".format(
                        model.__name__, scenario, optimisations, extra, dofit, prob, method_name))
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
                    execute=execute, location_pyspy=dest_pyspy_model,
                    patterns=patterns)
                for cr in created:
                    if cr in all_created:
                        raise RuntimeError(  # pragma: no cover
                            "File '{}' was already created.".format(cr))
                    all_created.add(cr)
                    if verbose > 1 and fLOG is not None:
                        fLOG("[create_asv_benchmark] add '{}'.".format(cr))
                    yield cr


def _create_asv_benchmark_file(  # pylint: disable=R0914
        location, model, scenario, optimisations, new_conv_options,
        extra, dofit, problem, runtime, X_train, X_test, y_train,
        y_test, Xort_test, init_types, conv_options,
        method_name, n_features, dims, opsets,
        output_index, predict_kwargs, prefix_import,
        exc, execute=False, location_pyspy=None, patterns=None):
    """
    Creates a benchmark file based in the information received
    through the argument. It uses template @see cl TemplateBenchmark.
    """
    if patterns is None:
        raise ValueError("Patterns list is empty.")  # pragma: no cover

    def format_conv_options(d_options, class_name):
        if d_options is None:
            return None
        res = {}
        for k, v in d_options.items():
            if isinstance(k, type):
                if "." + class_name + "'" in str(k):
                    res[class_name] = v
                    continue
                raise ValueError(  # pragma: no cover
                    "Class '{}', unable to format options {}".format(
                        class_name, d_options))
            res[k] = v
        return res

    def _nick_name_options(model, opts):
        # Shorten common onnx options, see _CommonAsvSklBenchmark._to_onnx.
        if opts is None:
            return opts  # pragma: no cover
        short_opts = shorten_onnx_options(model, opts)
        if short_opts is not None:
            return short_opts
        res = {}
        for k, v in opts.items():
            if hasattr(k, '__name__'):
                res["####" + k.__name__ + "####"] = v
            else:
                res[k] = v  # pragma: no cover
        return res

    def _make_simple_name(name):
        simple_name = name.replace("bench_", "").replace("_bench", "")
        simple_name = simple_name.replace("bench.", "").replace(".bench", "")
        simple_name = simple_name.replace(".", "-")
        repl = {'_': '', 'solverliblinear': 'liblinear'}
        for k, v in repl.items():
            simple_name = simple_name.replace(k, v)
        return simple_name

    def _optdict2string(opt):
        if isinstance(opt, str):
            return opt
        if isinstance(opt, list):
            raise TypeError(
                "Unable to process type %r." % type(opt))
        reps = {True: 1, False: 0, 'zipmap': 'zm',
                'optim': 'opt'}
        info = []
        for k, v in sorted(opt.items()):
            if isinstance(v, dict):
                v = _optdict2string(v)
            if k.startswith('####'):
                k = ''
            i = '{}{}'.format(reps.get(k, k), reps.get(v, v))
            info.append(i)
        return "-".join(info)

    runtimes_abb = {
        'scikit-learn': 'skl',
        'onnxruntime1': 'ort',
        'onnxruntime2': 'ort2',
        'python': 'pyrt',
        'python_compiled': 'pyrtc',
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
                dofit, conv_options, problem,
                shorten=True)
        except ValueError as e:  # pragma: no cover
            if exc:
                raise e
            warnings.warn(str(e))
            continue
        filename = name.replace(".", "_") + ".py"
        try:
            class_content = _select_pattern_problem(problem, patterns)
        except ValueError as e:
            if exc:
                raise e  # pragma: no cover
            warnings.warn(str(e))
            continue
        full_class_name = _asv_class_name(
            model, scenario, optimisation, extra,
            dofit, conv_options, problem,
            shorten=False)
        class_name = name.replace(
            "bench.", "").replace(".", "_") + "_bench"

        # n_features, N, runtimes
        rep = {
            "['skl', 'pyrtc', 'ort'],  # values for runtime": str(runtime),
            "[1, 10, 100, 1000, 10000],  # values for N": str(dims),
            "[4, 20],  # values for nf": str(n_features),
            "[get_opset_number_from_onnx()],  # values for opset": str(opsets),
            "['float', 'double'],  # values for dtype":
                "['float']" if '-64' not in problem else "['double']",
            "[None],  # values for optim": "%r" % nck_opts,
        }
        for k, v in rep.items():
            if k not in class_content:
                raise ValueError("Unable to find '{}'\n{}.".format(  # pragma: no cover
                    k, class_content))
            class_content = class_content.replace(k, v + ',')
        class_content = class_content.split(
            "def _create_model(self):")[0].strip("\n ")
        if "####" in class_content:
            class_content = class_content.replace(
                "'####", "").replace("####'", "")
        if "####" in class_content:
            raise RuntimeError(  # pragma: no cover
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
        atts.append("chk_method_name = %r" % method_name)
        atts.append("par_scenario = %r" % scenario)
        atts.append("par_problem = %r" % problem)
        atts.append("par_optimisation = %r" % optimisation)
        if not dofit:
            atts.append("par_dofit = False")
        if merged_options is not None and len(merged_options) > 0:
            atts.append("par_convopts = %r" % format_conv_options(
                conv_options, model.__name__))
        atts.append("par_full_test_name = %r" % full_class_name)

        simple_name = _make_simple_name(name)
        atts.append("benchmark_name = %r" % simple_name)
        atts.append("pretty_name = %r" % simple_name)

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
        except SyntaxError as e:  # pragma: no cover
            raise SyntaxError("Unable to compile model '{}'\n{}".format(
                model.__name__, class_content)) from e

        # Verifies missing imports.
        to_import, _ = verify_code(class_content, exc=False)
        try:
            miss = find_missing_sklearn_imports(to_import)
        except ValueError as e:  # pragma: no cover
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
        except SyntaxError as e:  # pragma: no cover
            raise SyntaxError("Unable to compile model '{}'\n{}".format(
                model.__name__,
                _display_code_lines(class_content))) from e

        # executes to check import
        if execute:
            try:
                exec(obj, globals(), locals())  # pylint: disable=W0122
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Unable to process class '{}' ('{}') a script due to '{}'\n{}".format(
                        model.__name__, filename, str(e),
                        _display_code_lines(class_content))) from e

        # Saves
        fullname = os.path.join(location, filename)
        names.append(fullname)
        with open(fullname, "w", encoding='utf-8') as f:
            f.write(class_content)

        if location_pyspy is not None:
            # adding configuration for pyspy
            class_name = re.compile(
                'class ([A-Za-z_0-9]+)[(]').findall(class_content)[0]
            fullname_pyspy = os.path.splitext(
                os.path.join(location_pyspy, filename))[0]
            pyfold = os.path.splitext(os.path.split(fullname)[-1])[0]

            dtypes = ['float', 'double'] if '-64' in problem else ['float']
            for dim in dims:
                for nf in n_features:
                    for opset in opsets:
                        for dtype in dtypes:
                            for opt in nck_opts:
                                tmpl = pyspy_template.replace(
                                    '__PATH__', location)
                                tmpl = tmpl.replace(
                                    '__CLASSNAME__', class_name)
                                tmpl = tmpl.replace('__PYFOLD__', pyfold)
                                opt = "" if opt == {} else opt

                                first = True
                                for rt in runtime:
                                    if first:
                                        tmpl += textwrap.dedent("""

                                        def profile0_{rt}(iter, cl, N, nf, opset, dtype, optim):
                                            return setup_profile0(iter, cl, '{rt}', N, nf, opset, dtype, optim)
                                        iter = profile0_{rt}(iter, cl, {dim}, {nf}, {opset}, '{dtype}', {opt})
                                        print(datetime.now(), "iter", iter)

                                        """).format(rt=rt, dim=dim, nf=nf, opset=opset,
                                                    dtype=dtype, opt="%r" % opt)
                                        first = False

                                    tmpl += textwrap.dedent("""

                                    def profile_{rt}(iter, cl, N, nf, opset, dtype, optim):
                                        return setup_profile(iter, cl, '{rt}', N, nf, opset, dtype, optim)
                                    profile_{rt}(iter, cl, {dim}, {nf}, {opset}, '{dtype}', {opt})
                                    print(datetime.now(), "iter", iter)

                                    """).format(rt=rt, dim=dim, nf=nf, opset=opset,
                                                dtype=dtype, opt="%r" % opt)

                                thename = "{n}_{dim}_{nf}_{opset}_{dtype}_{opt}.py".format(
                                    n=fullname_pyspy, dim=dim, nf=nf,
                                    opset=opset, dtype=dtype, opt=_optdict2string(opt))
                                with open(thename, 'w', encoding='utf-8') as f:
                                    f.write(tmpl)
                                names.append(thename)

                                ext = '.bat' if sys.platform.startswith(
                                    'win') else '.sh'
                                script = os.path.splitext(thename)[0] + ext
                                short = os.path.splitext(
                                    os.path.split(thename)[-1])[0]
                                with open(script, 'w', encoding='utf-8') as f:
                                    f.write('py-spy record --native --function --rate=10 -o {n}_fct.svg -- {py} {n}.py\n'.format(
                                        py=sys.executable, n=short))
                                    f.write('py-spy record --native --rate=10 -o {n}_line.svg -- {py} {n}.py\n'.format(
                                        py=sys.executable, n=short))

    return names
