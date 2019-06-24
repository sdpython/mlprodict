"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
import os
from time import perf_counter
from importlib import import_module
import pickle
from timeit import Timer
import numpy
import pandas
import onnx
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from sklearn.base import BaseEstimator
from sklearn.decomposition import SparseCoder
from sklearn.ensemble import VotingClassifier, AdaBoostRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier, ClassifierChain, RegressorChain
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeRegressor
from .onnx_inference import OnnxInference
from .. import __version__ as ort_version
from .validate_problems import _problems, find_suitable_problem


def to_onnx(model, X=None, name=None, initial_types=None,
            target_opset=None):
    """
    Converts a model using on :epkg:`sklearn-onnx`.

    @param      model           model to convert
    @param      X               training set (at least one row),
                                can be None, it is used to infered the
                                input types (*initial_types*)
    @param      initial_types   if *X* is None, then *initial_types* must be
                                defined
    @param      name            name of the produced model
    @param      target_opset    to do it with a different target opset
    @return                     converted model
    """
    from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
    from skl2onnx.algebra.type_helper import guess_initial_types
    from skl2onnx import convert_sklearn

    if isinstance(model, OnnxOperatorMixin):
        return model.to_onnx(X=X, name=name)
    if name is None:
        name = model.__class__.__name__
    initial_types = guess_initial_types(X, initial_types)
    return convert_sklearn(model, initial_types=initial_types, name=name,
                           target_opset=target_opset)


def get_opset_number_from_onnx():
    """
    Retuns the current :epkg:`onnx` opset
    based on the installed version of :epkg:`onnx`.
    """
    return onnx.defs.onnx_opset_version()


def sklearn_operators(subfolder=None):
    """
    Builds the list of operators from :epkg:`scikit-learn`.
    The function goes through the list of submodule
    and get the list of class which inherit from
    :epkg:`scikit-learn:base:BaseEstimator`.

    @param      subfolder   look into only one subfolder
    """
    found = []
    for sub in sklearn__all__:
        if subfolder is not None and sub != subfolder:
            continue
        try:
            mod = import_module("{0}.{1}".format("sklearn", sub))
        except ModuleNotFoundError:
            continue
        cls = getattr(mod, "__all__", None)
        if cls is None:
            cls = list(mod.__dict__)
        cls = [mod.__dict__[cl] for cl in cls]
        for cl in cls:
            try:
                issub = issubclass(cl, BaseEstimator)
            except TypeError:
                continue
            if cl.__name__ in {'Pipeline', 'ColumnTransformer',
                               'FeatureUnion', 'BaseEstimator'}:
                continue
            if (sub in {'calibration', 'dummy', 'manifold'} and
                    'Calibrated' not in cl.__name__):
                continue
            if issub:
                found.append(dict(name=cl.__name__, subfolder=sub, cl=cl))
    return found


def build_custom_scenarios():
    """
    Defines parameters values for some operators.

    .. runpython::
        :showcode:

        from mlprodict.onnxrt.validate import build_custom_scenarios
        import pprint
        pprint.pprint(build_custom_scenarios())
    """
    return {
        # skips
        SparseCoder: None,
        # scenarios
        AdaBoostRegressor: [
            ('default', {
                'n_estimators': 5,
            }),
        ],
        ClassifierChain: [
            ('logreg', {
                'base_estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        GridSearchCV: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
                'param_grid': {'fit_intercept': [False, True]},
            }),
            ('reg', {
                'estimator': LinearRegression(),
                'param_grid': {'fit_intercept': [False, True]},
            }),
        ],
        LocalOutlierFactor: [
            ('novelty', {
                'novelty': True,
            }),
        ],
        LogisticRegression: [
            ('liblinear', {
                'solver': 'liblinear',
            }),
        ],
        MultiOutputClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        MultiOutputRegressor: [
            ('linreg', {
                'estimator': LinearRegression(),
            })
        ],
        NuSVC: [
            ('prob', {
                'probability': True,
            }),
        ],
        OneVsOneClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        OneVsRestClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        OutputCodeClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        RandomizedSearchCV: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
                'param_distributions': {'fit_intercept': [False, True]},
            }),
            ('reg', {
                'estimator': LinearRegression(),
                'param_distributions': {'fit_intercept': [False, True]},
            }),
        ],
        RegressorChain: [
            ('linreg', {
                'base_estimator': LinearRegression(),
            })
        ],
        RFE: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
            }),
            ('reg', {
                'estimator': LinearRegression(),
            })
        ],
        RFECV: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
            }),
            ('reg', {
                'estimator': LinearRegression(),
            })
        ],
        SelectFromModel: [
            ('rf', {
                'estimator': DecisionTreeRegressor(),
            }),
        ],
        SGDClassifier: [
            ('log', {
                'loss': 'log',
            }),
        ],
        SVC: [
            ('prob', {
                'probability': True,
            }),
        ],
        VotingClassifier: [
            ('logreg-noflatten', {
                'voting': 'soft',
                'flatten_transform': False,
                'estimators': [
                    ('lr1', LogisticRegression(solver='liblinear')),
                    ('lr2', LogisticRegression(
                        solver='liblinear', fit_intercept=False)),
                ],
            })
        ],
        VotingRegressor: [
            ('linreg', {
                'estimators': [
                    ('lr1', LinearRegression()),
                    ('lr2', LinearRegression(fit_intercept=False)),
                ],
            })
        ],
    }


_extra_parameters = build_custom_scenarios()


def _measure_time(fct):
    """
    Measures the execution time for a function.
    """
    begin = perf_counter()
    res = fct()
    end = perf_counter()
    return res, end - begin


def _measure_absolute_difference(skl_pred, ort_pred):
    """
    *Measures the differences between predictions
    between two ways of computing them.
    The functions returns nan if shapes are different.
    """
    ort_pred_ = ort_pred
    if isinstance(ort_pred, list):
        if isinstance(ort_pred[0], dict):
            ort_pred = pandas.DataFrame(ort_pred).values
        elif (isinstance(ort_pred[0], list) and
                isinstance(ort_pred[0][0], dict)):
            if len(ort_pred) == 1:
                ort_pred = pandas.DataFrame(ort_pred[0]).values
            elif len(ort_pred[0]) == 1:
                ort_pred = pandas.DataFrame([o[0] for o in ort_pred]).values
            else:
                raise RuntimeError("Unable to compute differences between"
                                   "\n{}--------\n{}".format(
                                       skl_pred, ort_pred))
        else:
            ort_pred = numpy.array(ort_pred)

    if hasattr(skl_pred, 'todense'):
        skl_pred = skl_pred.todense()
    if hasattr(ort_pred, 'todense'):
        ort_pred = ort_pred.todense()

    if isinstance(ort_pred, list):
        raise RuntimeError("Issue with {}\n{}".format(ort_pred, ort_pred_))

    if skl_pred.shape != ort_pred.shape and skl_pred.size == ort_pred.size:
        ort_pred = ort_pred.ravel()
        skl_pred = skl_pred.ravel()

    if skl_pred.shape != ort_pred.shape:
        return 1e9

    diff = numpy.max(numpy.abs(skl_pred.ravel() - ort_pred.ravel()))

    if numpy.isnan(diff):
        raise RuntimeError("Unable to compute differences between {}-{}\n{}\n"
                           "--------\n{}".format(
                               skl_pred.shape, ort_pred.shape,
                               skl_pred, ort_pred))
    return diff


def _shape_exc(obj):
    if hasattr(obj, 'shape'):
        return obj.shape
    if isinstance(obj, (list, dict, tuple)):
        return "[{%d}]" % len(obj)
    return None


def dump_into_folder(dump_folder, obs_op=None, **kwargs):
    """
    Dumps information when an error was detected
    using :epkg:`*py:pickle`.

    @param      dump_folder     dump_folder
    @param      obs_op          obs_op (information)
    @kwargs                     kwargs
    """
    parts = (obs_op['runtime'], obs_op['name'], obs_op['scenario'],
             obs_op['problem'], obs_op.get('opset', '-'))
    name = "dump-ERROR-{}.pkl".format("-".join(map(str, parts)))
    name = os.path.join(dump_folder, name)
    obs_op = obs_op.copy()
    fcts = [k for k in obs_op if k.startswith('lambda')]
    for fct in fcts:
        del obs_op[fct]
    kwargs.update({'obs_op': obs_op})
    with open(name, "wb") as f:
        pickle.dump(kwargs, f)


def enumerate_compatible_opset(model, opset_min=9, opset_max=None,
                               check_runtime=True, debug=False,
                               runtime='CPU', dump_folder=None,
                               store_models=False, benchmark=False,
                               fLOG=print):
    """
    Lists all compatible opsets for a specific model.

    @param      model           operator class
    @param      opset_min       starts with this opset
    @param      opset_max       ends with this opset (None to use
                                current onnx opset)
    @param      check_runtime   checks that runtime can consume the
                                model and compute predictions
    @param      debug           catch exception (True) or not (False)
    @param      runtime         test a specific runtime, by default ``'CPU'``
    @param      dump_folder     dump information to replicate in case of mismatch
    @param      store_models    if True, the function
                                also stores the fitted model and its conversion
                                into :epkg:`ONNX`
    @param      benchmark       if True, measures the time taken by each function
                                to predict for different number of rows
    @param      fLOG            logging function
    @return                     dictionaries, each row has the following
                                keys: opset, exception if any, conversion time,
                                problem chosen to test the conversion...

    The function requires :epkg:`sklearn-onnx`.
    The outcome can be seen at pages references
    by :ref:`l-onnx-availability`.
    """
    try:
        problems = find_suitable_problem(model)
    except RuntimeError as e:
        yield {'name': model.__name__, 'skl_version': sklearn_version,
               '_0problem_exc': e}
        problems = []

    extras = _extra_parameters.get(model, [('default', {})])

    if opset_max is None:
        opset_max = get_opset_number_from_onnx()
    opsets = list(range(opset_min, opset_max + 1))
    opsets.append(None)

    if extras is None:
        problems = []
        yield {'name': model.__name__, 'skl_version': sklearn_version,
               '_0problem_exc': 'SKIPPED'}

    for prob in problems:
        X_, y_, init_types, method, output_index, Xort_ = _problems[prob]()
        if y_ is None:
            (X_train, X_test, Xort_train,  # pylint: disable=W0612
                Xort_test) = train_test_split(
                    X_, Xort_, random_state=42)
        else:
            (X_train, X_test, y_train, y_test,  # pylint: disable=W0612
                Xort_train, Xort_test) = train_test_split(
                    X_, y_, Xort_, random_state=42)

        for scenario, extra in extras:

            # training
            obs = {'scenario': scenario, 'name': model.__name__,
                   'skl_version': sklearn_version, 'problem': prob,
                   'method': method, 'output_index': output_index}
            try:
                inst = model(**extra)
            except TypeError as e:
                if debug:
                    raise
                import pprint
                raise RuntimeError(
                    "Unable to instantiate model '{}'.\nextra=\n{}".format(
                        model.__name__, pprint.pformat(extra))) from e

            try:
                if y_ is None:
                    t1 = _measure_time(lambda: inst.fit(X_train))[1]
                else:
                    t1 = _measure_time(lambda: inst.fit(X_train, y_train))[1]
            except (AttributeError, TypeError, ValueError, IndexError) as e:
                if debug:
                    raise
                obs["_1training_time_exc"] = str(e)
                yield obs
                continue

            obs["training_time"] = t1
            if store_models:
                obs['MODEL'] = inst
                obs['X_test'] = X_test
                obs['Xort_test'] = Xort_test
                obs['init_types'] = init_types

            # runtime
            if check_runtime:

                # compute sklearn prediction
                obs['ort_version'] = ort_version
                try:
                    meth = getattr(inst, method)
                except AttributeError as e:
                    if debug:
                        raise
                    obs['_2skl_meth_exc'] = str(e)
                    yield obs
                    continue
                try:
                    ypred, t4 = _measure_time(lambda: meth(X_test))
                    obs['lambda-skl'] = (lambda xo: meth(xo), X_test)
                except (ValueError, AttributeError, TypeError) as e:
                    if debug:
                        raise
                    obs['_3prediction_exc'] = str(e)
                    yield obs
                    continue
                obs['prediction_time'] = t4
                if benchmark and 'lambda-skl' in obs:
                    obs['bench-skl'] = benchmark_fct(*obs['lambda-skl'])

            # converting
            for opset in opsets:
                obs_op = obs.copy()
                if opset is not None:
                    obs_op['opset'] = opset

                if len(init_types) != 1:
                    raise NotImplementedError("Multiple types are is not implemented: "
                                              "{}.".format(init_types))

                def fct_skl(itt=inst, it=init_types[0][1], ops=opset):  # pylint: disable=W0102
                    return to_onnx(itt, it, target_opset=ops)

                try:
                    conv, t2 = _measure_time(fct_skl)
                    obs_op["convert_time"] = t2
                except RuntimeError as e:
                    if debug:
                        raise
                    obs_op["_4convert_exc"] = e
                    yield obs_op
                    continue

                if store_models:
                    obs_op['ONNX'] = conv

                # opset_domain
                for op_imp in list(conv.opset_import):
                    obs_op['domain_opset_%s' % op_imp.domain] = op_imp.version

                # prediction
                if check_runtime:
                    yield _call_runtime(obs_op=obs_op, conv=conv, opset=opset, debug=debug,
                                        runtime=runtime, inst=inst, X_=X_, y_=y_,
                                        init_types=init_types, method=method,
                                        output_index=output_index, Xort_=Xort_,
                                        ypred=ypred, Xort_test=Xort_test,
                                        model=model, dump_folder=dump_folder,
                                        benchmark=benchmark and opset == opsets[-1])
                else:
                    yield obs_op


def _call_runtime(obs_op, conv, opset, debug, inst, runtime,
                  X_, y_, init_types, method, output_index,
                  Xort_, ypred, Xort_test, model, dump_folder,
                  benchmark):
    """
    Private.
    """
    ser, t5 = _measure_time(lambda: conv.SerializeToString())
    obs_op['tostring_time'] = t5

    # load
    try:
        sess, t6 = _measure_time(
            lambda: OnnxInference(ser, runtime=runtime))
        obs_op['tostring_time'] = t6
    except (RuntimeError, ValueError) as e:
        if debug:
            raise
        obs_op['_5ort_load_exc'] = e
        return obs_op

    # compute batch
    def fct_batch(se=sess, xo=Xort_test, it=init_types):  # pylint: disable=W0102
        return se.run({it[0][0]: xo})
    try:
        opred, t7 = _measure_time(fct_batch)
        obs_op['ort_run_time_batch'] = t7
        obs_op['lambda-batch'] = (lambda xo: sess.run(
            {init_types[0][0]: xo}), Xort_test)
    except (RuntimeError, TypeError, ValueError, KeyError, IndexError) as e:
        if debug:
            raise
        obs_op['_6ort_run_batch_exc'] = e
    if benchmark and 'lambda-batch' in obs_op:
        obs_op['bench-batch'] = benchmark_fct(*obs_op['lambda-batch'])

    # difference
    if '_6ort_run_batch_exc' not in obs_op:
        if isinstance(opred, dict):
            ch = [(k, v) for k, v in sorted(opred.items())]
            # names = [_[0] for _ in ch]
            opred = [_[1] for _ in ch]

        try:
            opred = opred[output_index]
        except IndexError:
            if debug:
                raise
            obs_op['_8max_abs_diff_batch_exc'] = (
                "Unable to fetch output {}/{} for model '{}'"
                "".format(output_index, len(opred),
                          model.__name__))
            opred = None

        debug_exc = []
        if opred is not None:
            max_abs_diff = _measure_absolute_difference(
                ypred, opred)
            if numpy.isnan(max_abs_diff):
                obs_op['_8max_abs_diff_batch_exc'] = (
                    "Unable to compute differences between"
                    " {}-{}\n{}\n--------\n{}".format(
                        _shape_exc(
                            ypred), _shape_exc(opred),
                        ypred, opred))
                if debug:
                    debug_exc.append(RuntimeError(
                        obs_op['_8max_abs_diff_batch_exc']))
            else:
                obs_op['max_abs_diff_batch'] = max_abs_diff
                if dump_folder and max_abs_diff > 1e-5:
                    dump_into_folder(dump_folder, kind='batch', obs_op=obs_op,
                                     X_=X_, y_=y_, init_types=init_types,
                                     method=init_types, output_index=output_index,
                                     Xort_=Xort_)

    # compute single
    def fct_single(se=sess, xo=Xort_test, it=init_types):  # pylint: disable=W0102
        return [se.run({it[0][0]: Xort_row})
                for Xort_row in xo]
    try:
        opred, t7 = _measure_time(fct_single)
        obs_op['ort_run_time_single'] = t7
        obs_op['lambda-single'] = (
            lambda xo: [sess.run({init_types[0][0]: Xort_row})
                        for Xort_row in xo],
            Xort_test
        )
    except (RuntimeError, TypeError, ValueError, KeyError, IndexError) as e:
        if debug:
            raise
        obs_op['_9ort_run_single_exc'] = e
    if benchmark and 'lambda-single' in obs_op and 'lambda-batch' not in obs_op:
        obs_op['bench-single'] = benchmark_fct(*obs_op['lambda-single'])

    # difference
    if '_9ort_run_single_exc' not in obs_op:
        if isinstance(opred[0], dict):
            ch = [[(k, v) for k, v in sorted(o.items())]
                  for o in opred]
            # names = [[_[0] for _ in row] for row in ch]
            opred = [[_[1] for _ in row] for row in ch]

        try:
            opred = [o[output_index] for o in opred]
        except IndexError:
            if debug:
                raise
            obs_op['_Amax_abs_diff_single_exc'] = (
                "Unable to fetch output {}/{} for model '{}'"
                "".format(output_index, len(opred),
                          model.__name__))
            opred = None
        if opred is not None:
            max_abs_diff = _measure_absolute_difference(
                ypred, opred)
            if numpy.isnan(max_abs_diff):
                obs_op['_Amax_abs_diff_single_exc'] = (
                    "Unable to compute differences between"
                    "\n{}\n--------\n{}".format(
                        ypred, opred))
                if debug:
                    debug_exc.append(RuntimeError(
                        obs_op['_Amax_abs_diff_single_exc']))
            else:
                obs_op['max_abs_diff_single'] = max_abs_diff
                if dump_folder and max_abs_diff > 1e-5:
                    dump_into_folder(dump_folder, kind='single', obs_op=obs_op,
                                     X_=X_, y_=y_, init_types=init_types,
                                     method=init_types, output_index=output_index,
                                     Xort_=Xort_)

    if debug and len(debug_exc) == 2:
        raise debug_exc[0]
    if debug:
        import pprint
        pprint.pprint(obs_op)
    return obs_op


def enumerate_validated_operator_opsets(verbose=0, opset_min=9, opset_max=None,
                                        check_runtime=True, debug=False, runtime='CPU',
                                        models=None, dump_folder=None, store_models=False,
                                        benchmark=False, fLOG=print):
    """
    Tests all possible configuration for all possible
    operators and returns the results.

    @param      verbose         integer 0, 1, 2
    @param      opset_min       checks conversion starting from the opset
    @param      opset_max       checks conversion up to this opset,
                                None means @see fn get_opset_number_from_onnx.
    @param      check_runtime   checks the python runtime
    @param      models          only process a small list of operators,
                                set of model names
    @param      debug           stops whenever an exception
                                is raised
    @param      runtime         test a specific runtime, by default ``'CPU'``
    @param      dump_folder     dump information to replicate in case of mismatch
    @param      store_models    if True, the function
                                also stores the fitted model and its conversion
                                into :epkg:`ONNX`
    @param      benchmark       if True, measures the time taken by each function
                                to predict for different number of rows
    @param      fLOG            logging function
    @return                     list of dictionaries

    The function is available through command line
    :ref:`validate_runtime <l-cmd-validate_runtime>`.
    """
    ops = [_ for _ in sklearn_operators()]

    if models is not None:
        if not all(map(lambda m: isinstance(m, str), models)):
            raise ValueError("models must be a set of strings.")
        ops_ = [_ for _ in ops if _['name'] in models]
        if len(ops) == 0:
            raise ValueError("Parameter models is wrong: {}\n{}".format(
                models, ops[0]))
        ops = ops_

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
                from tqdm import tqdm
                loop = tqdm(ops)
            except ImportError:

                loop = iterate()
    else:
        loop = ops

    current_opset = get_opset_number_from_onnx()
    for row in loop:

        model = row['cl']

        for obs in enumerate_compatible_opset(
                model, opset_min=opset_min, opset_max=opset_max,
                check_runtime=check_runtime, runtime=runtime,
                debug=debug, dump_folder=dump_folder,
                store_models=store_models, benchmark=benchmark,
                fLOG=fLOG):

            if verbose > 1:
                fLOG("  ", obs)
            elif verbose > 0 and "_0problem_exc" in obs:
                fLOG("  ???", obs)

            diff = obs.get('max_abs_diff_batch',
                           obs.get('max_abs_diff_single', None))
            batch = 'max_abs_diff_batch' in obs and diff is not None
            op1 = obs.get('domain_opset_', '')
            op2 = obs.get('domain_opset_ai.onnx.ml', '')
            op = '{}|{}'.format(op1, op2)

            if diff is not None:
                if diff < 1e-5:
                    obs['available'] = 'OK'
                elif diff < 0.0001:
                    obs['available'] = 'e<0.0001'
                elif diff < 0.001:
                    obs['available'] = 'e<0.001'
                elif diff < 0.01:
                    obs['available'] = 'e<0.01'
                elif diff < 0.1:
                    obs['available'] = 'e<0.1'
                else:
                    obs['available'] = "ERROR->=%1.1f" % diff
                obs['available'] += '-' + op
                if not batch:
                    obs['available'] += "-NOBATCH"

            else:
                excs = []
                for k, v in sorted(obs.items()):
                    if k.endswith('_exc'):
                        excs.append((k, v))
                        break
                if 'opset' not in obs:
                    # It fails before the conversion happens.
                    obs['opset'] = current_opset
                if obs['opset'] == current_opset:
                    if len(excs) > 0:
                        k, v = excs[0]
                        obs['available'] = 'ERROR-%s' % k
                        obs['available-ERROR'] = v
                    else:
                        obs['available'] = 'ERROR-?'

            if 'bench-skl' in obs:
                b1 = obs['bench-skl']
                if 'bench-batch' in obs:
                    b2 = obs['bench-batch']
                elif 'bench-single' in obs:
                    b2 = obs['bench-single']
                else:
                    b2 = None
                if b1 is not None and b2 is not None:
                    for k in b1:
                        if k in b2 and b2[k] is not None and b1[k] is not None:
                            key = 'time-ratio-N=%d' % k
                            obs[key] = b2[k]['average'] / b1[k]['average']

            obs.update(row)
            yield obs


def summary_report(df):
    """
    Finalizes the results computed by function
    @see fn enumerate_validated_operator_opsets.

    @param      df      dataframe
    @return             pivoted dataframe

    The outcome can be seen at page about :ref:`l-onnx-pyrun`.
    """

    def aggfunc(values):
        if len(values) != 1:
            vals = set(values)
            if len(vals) != 1:
                return " // ".join(map(str, values))
        val = values.iloc[0]
        if isinstance(val, float) and numpy.isnan(val):
            return ""
        else:
            return val

    piv = pandas.pivot_table(df, values="available",
                             index=['name', 'problem', 'scenario'],
                             columns='opset',
                             aggfunc=aggfunc).reset_index(drop=False)

    opmin = min(df['opset'].dropna())
    versions = ["opset%d" % (opmin + t - 1)
                for t in range(1, piv.shape[1] - 2)]
    indices = ["name", "problem", "scenario"]
    piv.columns = indices + versions
    piv = piv[indices + list(reversed(versions))].copy()

    if "available-ERROR" in df.columns:

        from skl2onnx.common.exceptions import MissingShapeCalculator

        def replace_msg(text):
            if isinstance(text, MissingShapeCalculator):
                return "NO CONVERTER"
            if str(text).startswith("Unable to find a shape calculator for type '"):
                return "NO CONVERTER"
            return str(text)

        piv2 = pandas.pivot_table(df, values="available-ERROR",
                                  index=['name', 'problem', 'scenario'],
                                  columns='opset',
                                  aggfunc=aggfunc).reset_index(drop=False)

        col = piv2.iloc[:, piv2.shape[1] - 1]
        piv["ERROR-msg"] = col.apply(replace_msg)

    if "time-ratio-N=1" in df.columns:
        cols = [c for c in df.columns if c.startswith('time-ratio')]
        cols.sort()

        df_sub = df[['name', 'problem', 'scenario'] + cols]
        piv2 = df_sub.groupby(['name', 'problem', 'scenario']).mean()
        piv = piv.merge(piv2, on=['name', 'problem', 'scenario'], how='left')

        def rep(c):
            if 'N=1' in c and 'N=10' not in c:
                return c.replace("time-ratio-", "RT/SKL-")
            else:
                return c.replace("time-ratio-", "")
        cols = [rep(c) for c in piv.columns]
        piv.columns = cols

    def clean_values(value):
        if not isinstance(value, str):
            return value
        if "ERROR->=1000000" in value:
            value = "big-diff"
        elif "ERROR" in value:
            value = value.replace("ERROR-_", "")
            value = value.replace("_exc", "")
            value = "ERR: " + value
        elif "OK-" in value:
            value = value.replace("OK-", "OK ")
        elif "e<" in value:
            value = value.replace("-", " ")
        return value

    for c in piv.columns:
        if "opset" in c:
            piv[c] = piv[c].apply(clean_values)

    return piv


def measure_time(stmt, x, repeat=10, number=50, div_by_number=False):
    """
    Measures a statement and returns the results as a dictionary.

    @param      stmt            string
    @param      x               matrix
    @param      repeat          average over *repeat* experiment
    @param      number          number of executions in one row
    @param      div_by_number   divide by the number of executions
    @return                     dictionary

    See `Timer.repeat <https://docs.python.org/3/library/timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    if x is None:
        raise ValueError("x cannot be None")

    try:
        stmt(x)
    except RuntimeError as e:
        raise RuntimeError("{}-{}".format(type(x), x.dtype)) from e

    def fct():
        stmt(x)

    tim = Timer(fct)
    res = numpy.array(tim.repeat(repeat=repeat, number=number))
    total = numpy.sum(res)
    if div_by_number:
        res /= number
    mean = numpy.mean(res)
    dev = numpy.mean(res ** 2)
    dev = (dev - mean**2) ** 0.5
    mes = dict(average=mean, deviation=dev, min_exec=numpy.min(res),
               max_exec=numpy.max(res), repeat=repeat, number=number,
               total=total)
    return mes


def benchmark_fct(fct, X, time_limit=4):
    """
    Benchmarks a function which takes an array
    as an input and changes the number of rows.

    @param      fct         function to benchmark, signature
                            is fct(xo)
    @param      X           array
    @param      time_limit  above this time, measurement as stopped
    @return                 dictionary with the results
    """

    def make(x, n):
        if n < x.shape[0]:
            return x[:n].copy()
        else:
            r = numpy.empty((N, x.shape[1]), dtype=x.dtype)
            for i in range(0, N, x.shape[0]):
                end = min(i + x.shape[0], N)
                r[i: end, :] = x[0: end - i, :]
            return r

    res = {}
    for N in [1, 10, 100, 1000, 10000, 100000]:
        x = make(X, N)
        if N <= 10:
            repeat = 20
            number = 20
        elif N <= 1000:
            repeat = 5
            number = 5
        elif N <= 10000:
            repeat = 3
            number = 3
        else:
            repeat = 1
            number = 1
        res[N] = measure_time(fct, x, repeat=repeat,
                              number=number, div_by_number=True)
        if res[N] is not None and res[N].get('total', time_limit) >= time_limit:
            # too long
            break
    return res
