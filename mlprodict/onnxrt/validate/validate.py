"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
import pprint
from inspect import signature
import numpy
from numpy.linalg import LinAlgError
import sklearn
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from ... import __version__ as ort_version
from ...onnx_conv import to_onnx, register_converters, register_rewritten_operators
from ...tools.model_info import analyze_model, set_random_state
from ...tools.asv_options_helper import (
    get_opset_number_from_onnx, get_ir_version_from_onnx)
from ..onnx_inference import OnnxInference
from ..optim.sklearn_helper import inspect_sklearn_model, set_n_jobs
from ..optim.onnx_helper import onnx_statistics
from ..optim import onnx_optimisations
from .validate_problems import find_suitable_problem
from .validate_scenarios import _extra_parameters
from .validate_difference import measure_relative_difference
from .validate_helper import (
    _dispsimple, sklearn_operators,
    _measure_time, _shape_exc, dump_into_folder,
    default_time_kwargs, RuntimeBadResultsError,
    _dictionary2str, _merge_options, _multiply_time_kwargs,
    _get_problem_data)
from .validate_benchmark import benchmark_fct


@ignore_warnings(category=(UserWarning, ConvergenceWarning))
def _dofit_model(dofit, obs, inst, X_train, y_train, X_test, y_test,
                 Xort_test, init_types, store_models,
                 debug, verbose, fLOG):
    if dofit:
        if verbose >= 2 and fLOG is not None:
            fLOG("[enumerate_compatible_opset] fit, type: '{}' dtype: {}".format(
                type(X_train), getattr(X_train, 'dtype', '-')))
        try:
            set_random_state(inst)
            if y_train is None:
                t4 = _measure_time(lambda: inst.fit(X_train))[1]
            else:
                t4 = _measure_time(
                    lambda: inst.fit(X_train, y_train))[1]
        except (AttributeError, TypeError, ValueError,
                IndexError, NotImplementedError, MemoryError,
                LinAlgError, StopIteration) as e:
            if debug:
                raise  # pragma: no cover
            obs["_1training_time_exc"] = str(e)
            return False

        obs["training_time"] = t4
        try:
            skl_st = inspect_sklearn_model(inst)
        except NotImplementedError:
            skl_st = {}
        obs.update({'skl_' + k: v for k, v in skl_st.items()})

        if store_models:
            obs['MODEL'] = inst
            obs['X_test'] = X_test
            obs['Xort_test'] = Xort_test
            obs['init_types'] = init_types
    else:
        obs["training_time"] = 0.
        if store_models:
            obs['MODEL'] = inst
            obs['init_types'] = init_types

    return True


def _run_skl_prediction(obs, check_runtime, assume_finite, inst,
                        method_name, predict_kwargs, X_test,
                        benchmark, debug, verbose, time_kwargs,
                        skip_long_test, time_kwargs_fact, fLOG):
    if not check_runtime:
        return None  # pragma: no cover
    if verbose >= 2 and fLOG is not None:
        fLOG("[enumerate_compatible_opset] check_runtime SKL {}-{}-{}-{}-{}".format(
            id(inst), method_name, predict_kwargs, time_kwargs,
            time_kwargs_fact))
    with sklearn.config_context(assume_finite=assume_finite):
        # compute sklearn prediction
        obs['ort_version'] = ort_version
        try:
            meth = getattr(inst, method_name)
        except AttributeError as e:
            if debug:
                raise  # pragma: no cover
            obs['_2skl_meth_exc'] = str(e)
            return e
        try:
            ypred, t4, ___ = _measure_time(
                lambda: meth(X_test, **predict_kwargs))
            obs['lambda-skl'] = (lambda xo: meth(xo, **predict_kwargs), X_test)
        except (ValueError, AttributeError, TypeError, MemoryError, IndexError) as e:
            if debug:
                raise  # pragma: no cover
            obs['_3prediction_exc'] = str(e)
            return e
        obs['prediction_time'] = t4
        obs['assume_finite'] = assume_finite
        if benchmark and 'lambda-skl' in obs:
            obs['bench-skl'] = benchmark_fct(
                *obs['lambda-skl'], obs=obs,
                time_kwargs=_multiply_time_kwargs(
                    time_kwargs, time_kwargs_fact, inst),
                skip_long_test=skip_long_test)
        if verbose >= 3 and fLOG is not None:
            fLOG("[enumerate_compatible_opset] scikit-learn prediction")
            _dispsimple(ypred, fLOG)
        if verbose >= 2 and fLOG is not None:
            fLOG("[enumerate_compatible_opset] predictions stored")
    return ypred


def _retrieve_problems_extra(model, verbose, fLOG, extended_list):
    """
    Use by @see fn enumerate_compatible_opset.
    """
    extras = None
    if extended_list:
        from ...onnx_conv.validate_scenarios import find_suitable_problem as fsp_extended
        problems = fsp_extended(model)
        if problems is not None:
            from ...onnx_conv.validate_scenarios import build_custom_scenarios as fsp_scenarios
            extra_parameters = fsp_scenarios()

            if verbose >= 2 and fLOG is not None:
                fLOG(
                    "[enumerate_compatible_opset] found custom for model={}".format(model))
                extras = extra_parameters.get(model, None)
                if extras is not None:
                    fLOG(
                        "[enumerate_compatible_opset] found custom scenarios={}".format(extras))
    else:
        problems = None

    if problems is None:
        # scikit-learn
        extra_parameters = _extra_parameters
        try:
            problems = find_suitable_problem(model)
        except RuntimeError as e:
            return {'name': model.__name__, 'skl_version': sklearn_version,
                    '_0problem_exc': e}, extras
    extras = extra_parameters.get(model, [('default', {})])

    # checks existence of random_state
    sig = signature(model.__init__)
    if 'random_state' in sig.parameters:
        new_extras = []
        for extra in extras:
            if 'random_state' not in extra[1]:
                ps = extra[1].copy()
                ps['random_state'] = 42
                if len(extra) == 2:
                    extra = (extra[0], ps)
                else:
                    extra = (extra[0], ps) + extra[2:]
            new_extras.append(extra)
        extras = new_extras

    return problems, extras


def enumerate_compatible_opset(model, opset_min=-1, opset_max=-1,  # pylint: disable=R0914
                               check_runtime=True, debug=False,
                               runtime='python', dump_folder=None,
                               store_models=False, benchmark=False,
                               assume_finite=True, node_time=False,
                               fLOG=print, filter_exp=None,
                               verbose=0, time_kwargs=None,
                               extended_list=False, dump_all=False,
                               n_features=None, skip_long_test=True,
                               filter_scenario=None, time_kwargs_fact=None,
                               time_limit=4, n_jobs=None):
    """
    Lists all compatible opsets for a specific model.

    @param      model           operator class
    @param      opset_min       starts with this opset
    @param      opset_max       ends with this opset (None to use
                                current onnx opset)
    @param      check_runtime   checks that runtime can consume the
                                model and compute predictions
    @param      debug           catch exception (True) or not (False)
    @param      runtime         test a specific runtime, by default ``'python'``
    @param      dump_folder     dump information to replicate in case of mismatch
    @param      dump_all        dump all models not only the one which fail
    @param      store_models    if True, the function
                                also stores the fitted model and its conversion
                                into :epkg:`ONNX`
    @param      benchmark       if True, measures the time taken by each function
                                to predict for different number of rows
    @param      fLOG            logging function
    @param      filter_exp      function which tells if the experiment must be run,
                                None to run all, takes *model, problem* as an input
    @param      filter_scenario second function which tells if the experiment must be run,
                                None to run all, takes *model, problem, scenario, extra, options*
                                as an input
    @param      node_time       collect time for each node in the :epkg:`ONNX` graph
    @param      assume_finite   See `config_context
                                <https://scikit-learn.org/stable/modules/generated/
                                sklearn.config_context.html>`_, If True, validation for finiteness
                                will be skipped, saving time, but leading to potential crashes.
                                If False, validation for finiteness will be performed, avoiding error.
    @param      verbose         verbosity
    @param      extended_list   extends the list to custom converters
                                and problems
    @param      time_kwargs     to define a more precise way to measure a model
    @param      n_features      modifies the shorts datasets used to train the models
                                to use exactly this number of features, it can also
                                be a list to test multiple datasets
    @param      skip_long_test  skips tests for high values of N if they seem too long
    @param      time_kwargs_fact see :func:`_multiply_time_kwargs <mlprodict.onnxrt.validate.validate_helper._multiply_time_kwargs>`
    @param      time_limit      to stop benchmarking after this amount of time was spent
    @param      n_jobs          *n_jobs* is set to the number of CPU by default unless this
                                value is changed
    @return                     dictionaries, each row has the following
                                keys: opset, exception if any, conversion time,
                                problem chosen to test the conversion...

    The function requires :epkg:`sklearn-onnx`.
    The outcome can be seen at pages references
    by :ref:`l-onnx-availability`.
    The parameter *time_kwargs* is a dictionary which defines the
    number of times to repeat the same predictions in order
    to give more precise figures. The default value (if None) is returned
    by the following code:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.onnxrt.validate.validate_helper import default_time_kwargs
        import pprint
        pprint.pprint(default_time_kwargs())

    Parameter *time_kwargs_fact* multiples these values for some
    specific models. ``'lin'`` multiplies by 10 when the model
    is linear.
    """
    if opset_min == -1:
        opset_min = get_opset_number_from_onnx()  # pragma: no cover
    if opset_max == -1:
        opset_max = get_opset_number_from_onnx()  # pragma: no cover
    if verbose > 0 and fLOG is not None:
        fLOG("[enumerate_compatible_opset] opset in [{}, {}].".format(
            opset_min, opset_max))
    if verbose > 1 and fLOG:
        fLOG("[enumerate_compatible_opset] validate class '{}'.".format(
            model.__name__))
        if verbose > 2:
            fLOG(model)

    if time_kwargs is None:
        time_kwargs = default_time_kwargs()
    problems, extras = _retrieve_problems_extra(
        model, verbose, fLOG, extended_list)
    if isinstance(problems, dict):
        yield problems  # pragma: no cover
        problems = []  # pragma: no cover

    if opset_max is None:
        opset_max = get_opset_number_from_onnx()  # pragma: no cover
        opsets = list(range(opset_min, opset_max + 1))  # pragma: no cover
        opsets.append(None)  # pragma: no cover
    else:
        opsets = list(range(opset_min, opset_max + 1))

    if extras is None:
        problems = []
        yield {'name': model.__name__, 'skl_version': sklearn_version,
               '_0problem_exc': 'SKIPPED'}

    if not isinstance(n_features, list):
        n_features = [n_features]

    for prob in problems:
        if filter_exp is not None and not filter_exp(model, prob):
            continue
        for n_feature in n_features:
            if verbose >= 2 and fLOG is not None:
                fLOG("[enumerate_compatible_opset] problem={} n_feature={}".format(
                    prob, n_feature))

            (X_train, X_test, y_train,
             y_test, Xort_test,
             init_types, conv_options, method_name,
             output_index, dofit, predict_kwargs) = _get_problem_data(prob, n_feature)

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

                try:
                    scenario, extra = scenario_extra[:2]
                except TypeError as e:  # pragma: no cover
                    raise TypeError(
                        "Unable to interpret 'scenario_extra'\n{}".format(
                            scenario_extra)) from e
                if optimisations is None:
                    optimisations = [None]
                if new_conv_options is None:
                    new_conv_options = [{}]

                if (filter_scenario is not None and
                        not filter_scenario(model, prob, scenario,
                                            extra, new_conv_options)):
                    continue

                if verbose >= 2 and fLOG is not None:
                    fLOG("[enumerate_compatible_opset] ##############################")
                    fLOG("[enumerate_compatible_opset] scenario={} optim={} extra={} dofit={} (problem={})".format(
                        scenario, optimisations, extra, dofit, prob))

                # training
                obs = {'scenario': scenario, 'name': model.__name__,
                       'skl_version': sklearn_version, 'problem': prob,
                       'method_name': method_name, 'output_index': output_index,
                       'fit': dofit, 'conv_options': conv_options,
                       'idtype': Xort_test.dtype, 'predict_kwargs': predict_kwargs,
                       'init_types': init_types, 'inst': extra if extra else None,
                       'n_features': X_train.shape[1] if len(X_train.shape) == 2 else 1}
                inst = None
                extra = set_n_jobs(model, extra, n_jobs=n_jobs)
                try:
                    inst = model(**extra)
                except TypeError as e:  # pragma: no cover
                    if debug:  # pragma: no cover
                        raise
                    if "__init__() missing" not in str(e):
                        raise RuntimeError(
                            "Unable to instantiate model '{}'.\nextra=\n{}".format(
                                model.__name__, pprint.pformat(extra))) from e
                    yield obs.copy()
                    continue

                if not _dofit_model(dofit, obs, inst, X_train, y_train, X_test, y_test,
                                    Xort_test, init_types, store_models,
                                    debug, verbose, fLOG):
                    yield obs.copy()
                    continue

                # statistics about the trained model
                skl_infos = analyze_model(inst)
                for k, v in skl_infos.items():
                    obs['fit_' + k] = v

                # runtime
                ypred = _run_skl_prediction(
                    obs, check_runtime, assume_finite, inst,
                    method_name, predict_kwargs, X_test,
                    benchmark, debug, verbose, time_kwargs,
                    skip_long_test, time_kwargs_fact, fLOG)
                if isinstance(ypred, Exception):
                    yield obs.copy()
                    continue

                for run_obs in _call_conv_runtime_opset(
                        obs=obs.copy(), opsets=opsets, debug=debug,
                        new_conv_options=new_conv_options,
                        model=model, prob=prob, scenario=scenario,
                        extra=extra, extras=extras, conv_options=conv_options,
                        init_types=init_types, inst=inst,
                        optimisations=optimisations, verbose=verbose,
                        benchmark=benchmark, runtime=runtime,
                        filter_scenario=filter_scenario,
                        X_test=X_test, y_test=y_test, ypred=ypred,
                        Xort_test=Xort_test, method_name=method_name,
                        check_runtime=check_runtime,
                        output_index=output_index,
                        kwargs=dict(
                            dump_all=dump_all,
                            dump_folder=dump_folder,
                            node_time=node_time,
                            skip_long_test=skip_long_test,
                            store_models=store_models,
                            time_kwargs=_multiply_time_kwargs(
                                time_kwargs, time_kwargs_fact, inst)
                        ),
                        time_limit=time_limit,
                        fLOG=fLOG):
                    yield run_obs


def _check_run_benchmark(benchmark, stat_onnx, bench_memo, runtime):
    unique = set(stat_onnx.items())
    unique.add(runtime)
    run_benchmark = benchmark and all(
        map(lambda u: unique != u, bench_memo))
    if run_benchmark:
        bench_memo.append(unique)
    return run_benchmark


def _call_conv_runtime_opset(
        obs, opsets, debug, new_conv_options,
        model, prob, scenario, extra, extras, conv_options,
        init_types, inst, optimisations, verbose,
        benchmark, runtime, filter_scenario,
        check_runtime, X_test, y_test, ypred, Xort_test,
        method_name, output_index,
        kwargs, time_limit, fLOG):
    # Calls the conversion and runtime for different opets
    if None in opsets:
        set_opsets = [None] + list(sorted((_ for _ in opsets if _ is not None),
                                          reverse=True))
    else:
        set_opsets = list(sorted(opsets, reverse=True))
    bench_memo = []

    for opset in set_opsets:
        if verbose >= 2 and fLOG is not None:
            fLOG("[enumerate_compatible_opset] opset={} init_types={}".format(
                opset, init_types))
        obs_op = obs.copy()
        if opset is not None:
            obs_op['opset'] = opset

        if len(init_types) != 1:
            raise NotImplementedError(  # pragma: no cover
                "Multiple types are is not implemented: "
                "{}.".format(init_types))

        if not isinstance(runtime, list):
            runtime = [runtime]

        obs_op_0c = obs_op.copy()
        for aoptions in new_conv_options:
            obs_op = obs_op_0c.copy()
            all_conv_options = {} if conv_options is None else conv_options.copy()
            all_conv_options = _merge_options(
                all_conv_options, aoptions)
            obs_op['conv_options'] = all_conv_options

            if (filter_scenario is not None and
                    not filter_scenario(model, prob, scenario,
                                        extra, all_conv_options)):
                continue

            for rt in runtime:
                def fct_conv(itt=inst, it=init_types[0][1], ops=opset,
                             options=all_conv_options):
                    return to_onnx(itt, it, target_opset=ops, options=options,
                                   rewrite_ops=rt in ('', None, 'python',
                                                      'python_compiled'))

                if verbose >= 2 and fLOG is not None:
                    fLOG(
                        "[enumerate_compatible_opset] conversion to onnx: {}".format(all_conv_options))
                try:
                    conv, t4 = _measure_time(fct_conv)[:2]
                    obs_op["convert_time"] = t4
                except (RuntimeError, IndexError, AttributeError, TypeError,
                        ValueError, NameError, NotImplementedError) as e:
                    if debug:
                        fLOG(pprint.pformat(obs_op))  # pragma: no cover
                        raise  # pragma: no cover
                    obs_op["_4convert_exc"] = e
                    yield obs_op.copy()
                    continue

                if verbose >= 6 and fLOG is not None:
                    fLOG(  # pragma: no cover
                        "[enumerate_compatible_opset] ONNX:\n{}".format(conv))

                if all_conv_options.get('optim', '') == 'cdist':  # pragma: no cover
                    check_cdist = [_ for _ in str(conv).split('\n')
                                   if 'CDist' in _]
                    check_scan = [_ for _ in str(conv).split('\n')
                                  if 'Scan' in _]
                    if len(check_cdist) == 0 and len(check_scan) > 0:
                        raise RuntimeError(
                            "Operator CDist was not used in\n{}"
                            "".format(conv))

                obs_op0 = obs_op.copy()
                for optimisation in optimisations:
                    obs_op = obs_op0.copy()
                    if optimisation is not None:
                        if optimisation == 'onnx':
                            obs_op['optim'] = optimisation
                            if len(aoptions) != 0:
                                obs_op['optim'] += '/' + \
                                    _dictionary2str(aoptions)
                            conv = onnx_optimisations(conv)
                        else:
                            raise ValueError(  # pragma: no cover
                                "Unknown optimisation option '{}' (extra={})"
                                "".format(optimisation, extras))
                    else:
                        obs_op['optim'] = _dictionary2str(aoptions)

                    if verbose >= 3 and fLOG is not None:
                        fLOG("[enumerate_compatible_opset] optim='{}' optimisation={} all_conv_options={}".format(
                            obs_op['optim'], optimisation, all_conv_options))
                    if kwargs['store_models']:
                        obs_op['ONNX'] = conv
                        if verbose >= 2 and fLOG is not None:
                            fLOG(  # pragma: no cover
                                "[enumerate_compatible_opset] onnx nodes: {}".format(
                                    len(conv.graph.node)))
                    stat_onnx = onnx_statistics(conv)
                    obs_op.update(
                        {'onx_' + k: v for k, v in stat_onnx.items()})

                    # opset_domain
                    for op_imp in list(conv.opset_import):
                        obs_op['domain_opset_%s' %
                               op_imp.domain] = op_imp.version

                    run_benchmark = _check_run_benchmark(
                        benchmark, stat_onnx, bench_memo, rt)

                    # prediction
                    if check_runtime:
                        yield _call_runtime(obs_op=obs_op.copy(), conv=conv,
                                            opset=opset, debug=debug,
                                            runtime=rt, inst=inst,
                                            X_test=X_test, y_test=y_test,
                                            init_types=init_types,
                                            method_name=method_name,
                                            output_index=output_index,
                                            ypred=ypred, Xort_test=Xort_test,
                                            model=model,
                                            dump_folder=kwargs['dump_folder'],
                                            benchmark=run_benchmark,
                                            node_time=kwargs['node_time'],
                                            time_kwargs=kwargs['time_kwargs'],
                                            fLOG=fLOG, verbose=verbose,
                                            store_models=kwargs['store_models'],
                                            dump_all=kwargs['dump_all'],
                                            skip_long_test=kwargs['skip_long_test'],
                                            time_limit=time_limit)
                    else:
                        yield obs_op.copy()  # pragma: no cover


def _call_runtime(obs_op, conv, opset, debug, inst, runtime,
                  X_test, y_test, init_types, method_name, output_index,
                  ypred, Xort_test, model, dump_folder,
                  benchmark, node_time, fLOG,
                  verbose, store_models, time_kwargs,
                  dump_all, skip_long_test, time_limit):
    """
    Private.
    """
    if 'onnxruntime' in runtime:
        old = conv.ir_version
        conv.ir_version = get_ir_version_from_onnx()
    else:
        old = None

    ser, t5, ___ = _measure_time(lambda: conv.SerializeToString())
    obs_op['tostring_time'] = t5
    obs_op['runtime'] = runtime

    if old is not None:
        conv.ir_version = old

    # load
    if verbose >= 2 and fLOG is not None:
        fLOG("[enumerate_compatible_opset-R] load onnx")
    try:
        sess, t5, ___ = _measure_time(
            lambda: OnnxInference(ser, runtime=runtime))
        obs_op['tostring_time'] = t5
    except (RuntimeError, ValueError, KeyError, IndexError, TypeError) as e:
        if debug:
            raise  # pragma: no cover
        obs_op['_5ort_load_exc'] = e
        return obs_op

    # compute batch
    if store_models:
        obs_op['OINF'] = sess
    if verbose >= 2 and fLOG is not None:
        fLOG("[enumerate_compatible_opset-R] compute batch with runtime "
             "'{}'".format(runtime))

    def fct_batch(se=sess, xo=Xort_test, it=init_types):  # pylint: disable=W0102
        return se.run({it[0][0]: xo},
                      verbose=max(verbose - 1, 1) if debug else 0, fLOG=fLOG)

    try:
        opred, t5, ___ = _measure_time(fct_batch)
        obs_op['ort_run_time_batch'] = t5
        obs_op['lambda-batch'] = (lambda xo: sess.run(
            {init_types[0][0]: xo}, node_time=node_time), Xort_test)
    except (RuntimeError, TypeError, ValueError, KeyError, IndexError) as e:
        if debug:
            raise RuntimeError("Issue with {}.".format(
                obs_op)) from e  # pragma: no cover
        obs_op['_6ort_run_batch_exc'] = e
    if (benchmark or node_time) and 'lambda-batch' in obs_op:
        try:
            benres = benchmark_fct(*obs_op['lambda-batch'], obs=obs_op,
                                   node_time=node_time, time_kwargs=time_kwargs,
                                   skip_long_test=skip_long_test,
                                   time_limit=time_limit)
            obs_op['bench-batch'] = benres
        except (RuntimeError, TypeError, ValueError) as e:  # pragma: no cover
            if debug:
                raise e  # pragma: no cover
            obs_op['_6ort_run_batch_exc'] = e
            obs_op['_6ort_run_batch_bench_exc'] = e

    # difference
    debug_exc = []
    if verbose >= 2 and fLOG is not None:
        fLOG("[enumerate_compatible_opset-R] differences")
    if '_6ort_run_batch_exc' not in obs_op:
        if isinstance(opred, dict):
            ch = [(k, v) for k, v in opred.items()]
            opred = [_[1] for _ in ch]

        if output_index != 'all':
            try:
                opred = opred[output_index]
            except IndexError as e:  # pragma: no cover
                if debug:
                    raise IndexError(
                        "Issue with output_index={}/{}".format(
                            output_index, len(opred))) from e
                obs_op['_8max_rel_diff_batch_exc'] = (
                    "Unable to fetch output {}/{} for model '{}'"
                    "".format(output_index, len(opred),
                              model.__name__))
                opred = None

        if opred is not None:
            if store_models:
                obs_op['skl_outputs'] = ypred
                obs_op['ort_outputs'] = opred
            if verbose >= 3 and fLOG is not None:
                fLOG("[_call_runtime] runtime prediction")
                _dispsimple(opred, fLOG)

            if (method_name == "decision_function" and hasattr(opred, 'shape') and
                    hasattr(ypred, 'shape') and len(opred.shape) == 2 and
                    opred.shape[1] == 2 and len(ypred.shape) == 1):
                # decision_function, for binary classification,
                # raw score is a distance
                max_rel_diff = measure_relative_difference(
                    ypred, opred[:, 1])
            else:
                max_rel_diff = measure_relative_difference(
                    ypred, opred)

            if max_rel_diff >= 1e9 and debug:  # pragma: no cover
                _shape = lambda o: o.shape if hasattr(
                    o, 'shape') else 'no shape'
                raise RuntimeError(
                    "Big difference (opset={}, runtime='{}' p='{}' s='{}')"
                    ":\n-------\n{}-{}\n{}\n--------\n{}-{}\n{}".format(
                        opset, runtime, obs_op['problem'], obs_op['scenario'],
                        type(ypred), _shape(ypred), ypred,
                        type(opred), _shape(opred), opred))

            if numpy.isnan(max_rel_diff):
                obs_op['_8max_rel_diff_batch_exc'] = (  # pragma: no cover
                    "Unable to compute differences between"
                    " {}-{}\n{}\n--------\n{}".format(
                        _shape_exc(
                            ypred), _shape_exc(opred),
                        ypred, opred))
                if debug:  # pragma: no cover
                    debug_exc.append(RuntimeError(
                        obs_op['_8max_rel_diff_batch_exc']))
            else:
                obs_op['max_rel_diff_batch'] = max_rel_diff
                if dump_folder and max_rel_diff > 1e-5:
                    dump_into_folder(dump_folder, kind='batch', obs_op=obs_op,
                                     X_test=X_test, y_test=y_test, Xort_test=Xort_test)
                if debug and max_rel_diff >= 0.1:  # pragma: no cover
                    raise RuntimeError("Two big differences {}\n{}\n{}\n{}".format(
                        max_rel_diff, inst, conv, pprint.pformat(obs_op)))

    if debug and len(debug_exc) == 2:
        raise debug_exc[0]  # pragma: no cover
    if debug and verbose >= 2:  # pragma: no cover
        if verbose >= 3:
            fLOG(pprint.pformat(obs_op))
        else:
            obs_op_log = {k: v for k,
                          v in obs_op.items() if 'lambda-' not in k}
            fLOG(pprint.pformat(obs_op_log))
    if verbose >= 2 and fLOG is not None:
        fLOG("[enumerate_compatible_opset-R] next...")
    if dump_all:
        dump = dump_into_folder(dump_folder, kind='batch', obs_op=obs_op,
                                X_test=X_test, y_test=y_test, Xort_test=Xort_test,
                                is_error=len(debug_exc) > 1,
                                onnx_bytes=conv.SerializeToString(),
                                skl_model=inst, ypred=ypred)
        obs_op['dumped'] = dump
    return obs_op


def _enumerate_validated_operator_opsets_ops(extended_list, models, skip_models):
    ops = [_ for _ in sklearn_operators(extended=extended_list)]

    if models is not None:
        if not all(map(lambda m: isinstance(m, str), models)):
            raise ValueError(  # pragma: no cover
                "models must be a set of strings.")
        ops_ = [_ for _ in ops if _['name'] in models]
        if len(ops) == 0:
            raise ValueError(  # pragma: no cover
                "Parameter models is wrong: {}\n{}".format(
                    models, ops[0]))
        ops = ops_
    if skip_models is not None:
        ops = [m for m in ops if m['name'] not in skip_models]
    return ops


def _enumerate_validated_operator_opsets_version(runtime):
    from numpy import __version__ as numpy_version
    from onnx import __version__ as onnx_version
    from scipy import __version__ as scipy_version
    from skl2onnx import __version__ as skl2onnx_version
    add_versions = {'v_numpy': numpy_version, 'v_onnx': onnx_version,
                    'v_scipy': scipy_version, 'v_skl2onnx': skl2onnx_version,
                    'v_sklearn': sklearn_version, 'v_onnxruntime': ort_version}
    if "onnxruntime" in runtime:
        from onnxruntime import __version__ as onnxrt_version
        add_versions['v_onnxruntime'] = onnxrt_version
    return add_versions


def enumerate_validated_operator_opsets(verbose=0, opset_min=-1, opset_max=-1,
                                        check_runtime=True, debug=False, runtime='python',
                                        models=None, dump_folder=None, store_models=False,
                                        benchmark=False, skip_models=None,
                                        assume_finite=True, node_time=False,
                                        fLOG=print, filter_exp=None,
                                        versions=False, extended_list=False,
                                        time_kwargs=None, dump_all=False,
                                        n_features=None, skip_long_test=True,
                                        fail_bad_results=False,
                                        filter_scenario=None,
                                        time_kwargs_fact=None,
                                        time_limit=4, n_jobs=None):
    """
    Tests all possible configurations for all possible
    operators and returns the results.

    @param      verbose         integer 0, 1, 2
    @param      opset_min       checks conversion starting from the opset, -1
                                to get the last one
    @param      opset_max       checks conversion up to this opset,
                                None means @see fn get_opset_number_from_onnx.
    @param      check_runtime   checks the python runtime
    @param      models          only process a small list of operators,
                                set of model names
    @param      debug           stops whenever an exception
                                is raised
    @param      runtime         test a specific runtime, by default ``'python'``
    @param      dump_folder     dump information to replicate in case of mismatch
    @param      dump_all        dump all models not only the one which fail
    @param      store_models    if True, the function
                                also stores the fitted model and its conversion
                                into :epkg:`ONNX`
    @param      benchmark       if True, measures the time taken by each function
                                to predict for different number of rows
    @param      filter_exp      function which tells if the experiment must be run,
                                None to run all, takes *model, problem* as an input
    @param      filter_scenario second function which tells if the experiment must be run,
                                None to run all, takes *model, problem, scenario, extra, options*
                                as an input
    @param      skip_models     models to skip
    @param      assume_finite   See `config_context
                                <https://scikit-learn.org/stable/modules/generated/
                                sklearn.config_context.html>`_, If True, validation for finiteness
                                will be skipped, saving time, but leading to potential crashes.
                                If False, validation for finiteness will be performed, avoiding error.
    @param      node_time       measure time execution for every node in the graph
    @param      versions        add columns with versions of used packages,
                                :epkg:`numpy`, :epkg:`scikit-learn`, :epkg:`onnx`,
                                :epkg:`onnxruntime`, :epkg:`sklearn-onnx`
    @param      extended_list   also check models this module implements a converter for
    @param      time_kwargs     to define a more precise way to measure a model
    @param      n_features      modifies the shorts datasets used to train the models
                                to use exactly this number of features, it can also
                                be a list to test multiple datasets
    @param      skip_long_test  skips tests for high values of N if they seem too long
    @param      fail_bad_results fails if the results are aligned with :epkg:`scikit-learn`
    @param      time_kwargs_fact see :func:`_multiply_time_kwargs <mlprodict.onnxrt.validate.validate_helper._multiply_time_kwargs>`
    @param      time_limit      to skip the rest of the test after this limit (in second)
    @param      n_jobs          *n_jobs* is set to the number of CPU by default unless this
                                value is changed
    @param      fLOG            logging function
    @return                     list of dictionaries

    The function is available through command line
    :ref:`validate_runtime <l-cmd-validate_runtime>`.
    The default for *time_kwargs* is the following:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.onnxrt.validate.validate_helper import default_time_kwargs
        import pprint
        pprint.pprint(default_time_kwargs())
    """
    register_converters()
    register_rewritten_operators()
    ops = _enumerate_validated_operator_opsets_ops(
        extended_list, models, skip_models)

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

    if versions:
        add_versions = _enumerate_validated_operator_opsets_version(runtime)
    else:
        add_versions = {}

    current_opset = get_opset_number_from_onnx()
    if opset_min == -1:
        opset_min = get_opset_number_from_onnx()
    if opset_max == -1:
        opset_max = get_opset_number_from_onnx()
    if verbose > 0 and fLOG is not None:
        fLOG("[enumerate_validated_operator_opsets] opset in [{}, {}].".format(
            opset_min, opset_max))
    for row in loop:

        model = row['cl']
        if verbose > 1:
            fLOG("[enumerate_validated_operator_opsets] - model='{}'".format(model))

        for obs in enumerate_compatible_opset(
                model, opset_min=opset_min, opset_max=opset_max,
                check_runtime=check_runtime, runtime=runtime,
                debug=debug, dump_folder=dump_folder,
                store_models=store_models, benchmark=benchmark,
                fLOG=fLOG, filter_exp=filter_exp,
                assume_finite=assume_finite, node_time=node_time,
                verbose=verbose, extended_list=extended_list,
                time_kwargs=time_kwargs, dump_all=dump_all,
                n_features=n_features, skip_long_test=skip_long_test,
                filter_scenario=filter_scenario,
                time_kwargs_fact=time_kwargs_fact,
                time_limit=time_limit, n_jobs=n_jobs):

            for mandkey in ('inst', 'method_name', 'problem',
                            'scenario'):
                if '_0problem_exc' in obs:
                    continue
                if mandkey not in obs:
                    raise ValueError("Missing key '{}' in\n{}".format(
                        mandkey, pprint.pformat(obs)))  # pragma: no cover
            if verbose > 1:
                fLOG('[enumerate_validated_operator_opsets] - OBS')
                if verbose > 2:
                    fLOG("  ", obs)
                else:
                    obs_log = {k: v for k,
                               v in obs.items() if 'lambda-' not in k}
                    fLOG("  ", obs_log)
            elif verbose > 0 and "_0problem_exc" in obs:
                fLOG("  ???", obs)  # pragma: no cover

            diff = obs.get('max_rel_diff_batch', None)
            batch = 'max_rel_diff_batch' in obs and diff is not None
            op1 = obs.get('domain_opset_', '')
            op2 = obs.get('domain_opset_ai.onnx.ml', '')
            op = '{}/{}'.format(op1, op2)

            obs['available'] = "?"
            if diff is not None:
                if diff < 1e-5:
                    obs['available'] = 'OK'
                elif diff < 0.0001:
                    obs['available'] = 'e<0.0001'
                elif diff < 0.001:
                    obs['available'] = 'e<0.001'
                elif diff < 0.01:
                    obs['available'] = 'e<0.01'  # pragma: no cover
                elif diff < 0.1:
                    obs['available'] = 'e<0.1'
                else:
                    obs['available'] = "ERROR->=%1.1f" % diff
                obs['available'] += '-' + op
                if not batch:
                    obs['available'] += "-NOBATCH"  # pragma: no cover
                if fail_bad_results and 'e<' in obs['available']:
                    raise RuntimeBadResultsError(
                        "Wrong results '{}'.".format(obs['available']), obs)  # pragma: no cover

            excs = []
            for k, v in sorted(obs.items()):
                if k.endswith('_exc'):
                    excs.append((k, v))
                    break
            if 'opset' not in obs:
                # It fails before the conversion happens.
                obs['opset'] = current_opset
            if obs['opset'] == current_opset and len(excs) > 0:
                k, v = excs[0]
                obs['available'] = 'ERROR-%s' % k
                obs['available-ERROR'] = v

            if 'bench-skl' in obs:
                b1 = obs['bench-skl']
                if 'bench-batch' in obs:
                    b2 = obs['bench-batch']
                else:
                    b2 = None
                if b1 is not None and b2 is not None:
                    for k in b1:
                        if k in b2 and b2[k] is not None and b1[k] is not None:
                            key = 'time-ratio-N=%d' % k
                            obs[key] = b2[k]['average'] / b1[k]['average']
                            key = 'time-ratio-N=%d-min' % k
                            obs[key] = b2[k]['min_exec'] / b1[k]['max_exec']
                            key = 'time-ratio-N=%d-max' % k
                            obs[key] = b2[k]['max_exec'] / b1[k]['min_exec']

            obs.update(row)
            obs.update(add_versions)
            yield obs.copy()
