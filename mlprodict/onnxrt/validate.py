"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
from time import perf_counter
from importlib import import_module
import warnings
import numpy
import pandas
import onnx
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.decomposition import SparseCoder
from sklearn.ensemble import VotingClassifier, AdaBoostRegressor, VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier, ClassifierChain, RegressorChain
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.testing import ignore_warnings
from .onnx_inference import OnnxInference
from .. import __version__ as ort_version


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


def sklearn_operators():
    """
    Builds the list of operators from :epkg:`scikit-learn`.
    The function goes through the list of submodule
    and get the list of class which inherit from
    :epkg:`scikit-learn:base:BaseEstimator`.
    """
    found = []
    for sub in sklearn__all__:
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


def _problem_for_predictor_binary_classification():
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    binary classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    y[y == 2] = 1
    return (X, y, [('X', X[:1].astype(numpy.float32))],
            'predict_proba', 1, X.astype(numpy.float32))


def _problem_for_predictor_multi_classification():
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    multi-class classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y, [('X', X[:1].astype(numpy.float32))],
            'predict_proba', 1, X.astype(numpy.float32))


def _problem_for_predictor_regression():
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    multi-class classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y.astype(float), [('X', X[:1].astype(numpy.float32))],
            'predict', 0, X.astype(numpy.float32))


def _problem_for_numerical_transform():
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    multi-class classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', X[:1].astype(numpy.float32))],
            'transform', 0, X.astype(numpy.float32))


def find_suitable_problem(model):
    """
    Determines problems suitable for a given
    :epkg:`scikit-learn` operator. It may be
    * `bin-class`: binary classification
    * `mutli-class`: multi-class classification
    * `regression`: regression
    * `num-transform`: transform numerical features
    """
    if model in {RFE, RFECV, GridSearchCV}:
        return ['bin-class', 'multi-class', 'regression']
    if hasattr(model, 'predict_proba'):
        if model is OneVsRestClassifier:
            return ['multi-class']
        else:
            return ['bin-class', 'multi-class']

    if hasattr(model, 'predict'):
        return ['regression']

    if hasattr(model, 'transform'):
        return ['num-transform']

    raise RuntimeError("Unable to find problem for model '{}'."
                       "".format(model.__name__))


_problems = {
    "bin-class": _problem_for_predictor_binary_classification,
    "multi-class": _problem_for_predictor_multi_classification,
    "regression": _problem_for_predictor_regression,
    "num-transform": _problem_for_numerical_transform,
}


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
            ('logreg', {
                'voting': 'soft',
                'estimators': [
                    ('lr1', LogisticRegression(solver='liblinear')),
                    ('lr2', LogisticRegression(
                        solver='liblinear', fit_intercept=False)),
                ],
            })
        ],
        VotingRegressor: [
            ('logreg', {
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
    return res, begin - end


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
        warnings.warn("Unable to compute differences between {}-{}\n{}\n"
                      "--------\n{}".format(
                          skl_pred.shape, ort_pred.shape,
                          skl_pred, ort_pred))
        return numpy.nan

    diff = numpy.max(numpy.abs(skl_pred.ravel() - ort_pred.ravel()))

    if numpy.isnan(diff):
        raise RuntimeError("Unable to compute differences between {}-{}\n{}\n"
                           "--------\n{}".format(
                               skl_pred.shape, ort_pred.shape,
                               skl_pred, ort_pred))
    return diff


def enumerate_compatible_opset(model, opset_min=9, opset_max=None,
                               check_runtime=True, debug=False,
                               runtime='CPU', fLOG=print):
    """
    Lists all compatiable opsets for a specific model.

    @param      model           operator class
    @param      opset_min       starts with this opset
    @param      opset_max       ends with this opset (None to use
                                current onnx opset)
    @param      check_runtime   checks that runtime can consume the
                                model and compute predictions
    @param      debug           catch exception (True) or not (False)
    @param      runtime         test a specific runtime, by default ``'CPU'``
    @param      fLOG            logging function
    @return                     dictionaries, each row has the following
                                keys: opset, exception if any, conversion time,
                                problem chosen to test the conversion...

    The function requires :epkg:`sklearn-onnx`.
    The outcome can be seen at page about :ref:`l-onnx-pyrun`.
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
                   'skl_version': sklearn_version, 'problem': prob}
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
                except (ValueError, AttributeError) as e:
                    if debug:
                        raise
                    obs['_3prediction_exc'] = str(e)
                    yield obs
                    continue
                obs['prediction_time'] = t4

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

                # opset_domain
                for op_imp in list(conv.opset_import):
                    obs_op['domain_opset_%s' % op_imp.domain] = op_imp.version

                # prediction
                if check_runtime:
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
                        yield obs_op
                        continue

                    # compute batch
                    def fct_batch(se=sess, xo=Xort_test, it=init_types):  # pylint: disable=W0102
                        return se.run({it[0][0]: xo})
                    try:
                        opred, t7 = _measure_time(fct_batch)
                        obs_op['ort_run_time_batch'] = t7
                    except (RuntimeError, TypeError, ValueError) as e:
                        if debug:
                            raise
                        obs_op['_6ort_run_batch_exc'] = e

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
                        if opred is not None:
                            max_abs_diff = _measure_absolute_difference(
                                ypred, opred)
                            if debug and numpy.isnan(max_abs_diff):
                                raise RuntimeError(
                                    "Unable to compute differences between"
                                    " {}-{}\n{}\n--------\n{}".format(
                                        ypred.shape, opred.shape,
                                        ypred, opred))
                            obs_op['max_abs_diff_batch'] = max_abs_diff

                    # compute single
                    def fct_single(se=sess, xo=Xort_test, it=init_types):  # pylint: disable=W0102
                        return [se.run({it[0][0]: Xort_row})
                                for Xort_row in xo]
                    try:
                        opred, t7 = _measure_time(fct_single)
                        obs_op['ort_run_time_single'] = t7
                    except (RuntimeError, TypeError, ValueError) as e:
                        if debug:
                            raise
                        obs_op['_9ort_run_single_exc'] = e

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
                            if debug and numpy.isnan(max_abs_diff):
                                raise RuntimeError(
                                    "Unable to compute differences between"
                                    "\n{}\n--------\n{}".format(
                                        ypred, opred))
                            obs_op['max_abs_diff_single'] = max_abs_diff

                    if debug:
                        import pprint
                        pprint.pprint(obs_op)
                    yield obs_op


@ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
def validate_operator_opsets(verbose=0, opset_min=9, opset_max=None,
                             check_runtime=True, debug=None, runtime='CPU',
                             fLOG=print):
    """
    Tests all possible configuration for all possible
    operators and returns the results.

    @param      verbose         integer 0, 1, 2
    @param      opset_min       checks conversion starting from the opset
    @param      opset_max       checks conversion up to this opset,
                                None means @see fn get_opset_number_from_onnx.
    @param      check_runtime   checks the python runtime
    @param      debug           only checks a small list of operators,
                                set of model names
    @param      runtime         test a specific runtime, by default ``'CPU'``
    @param      fLOG            logging function
    @return                     list of dictionaries
    """
    ops = [_ for _ in sklearn_operators()]

    if debug is not None:
        ops_ = [_ for _ in ops if _['name'] in debug]
        if len(ops) == 0:
            raise ValueError("Debug is wrong: {}\n{}".format(
                debug, ops[0]))
        ops = ops_

    if verbose > 0:
        try:
            from tqdm import tqdm
            loop = tqdm(ops)
        except ImportError:

            def iterate():
                for i, row in enumerate(ops):
                    fLOG("{}/{} - {}".format(i + 1, len(ops), row))
                    yield row

            loop = iterate()
    else:
        loop = ops

    current_opset = get_opset_number_from_onnx()
    rows = []
    for row in loop:

        model = row['cl']

        for obs in enumerate_compatible_opset(
                model, opset_min=opset_min, opset_max=opset_max,
                check_runtime=check_runtime, runtime=runtime,
                debug=debug is not None, fLOG=fLOG):
            if verbose > 1:
                fLOG("  ", obs)
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
            elif 'opset' in obs and obs['opset'] == current_opset:
                excs = []
                for k, v in sorted(obs.items()):
                    if k.endswith('_exc'):
                        excs.append((k, v))
                        break
                if len(excs) > 0:
                    k, v = excs[0]
                    obs['available'] = 'ERROR-%s' % k
                    obs['available-ERROR'] = v
                else:
                    obs['available'] = 'ERROR-?'
            obs.update(row)
            rows.append(obs)

    return rows


def summary_report(df):
    """
    Finalizes the results computed by function
    @see fn validate_operator_opsets.

    @param      df      dataframe
    @return             pivoted dataframe

    The outcome can be seen at page about :ref:`l-onnx-pyrun`.
    """

    def aggfunc(values):
        if len(values) != 1:
            raise ValueError(values)
        val = values.iloc[0]
        if isinstance(val, float) and numpy.isnan(val):
            return ""
        else:
            return val

    piv = pandas.pivot_table(df, values="available",
                             index=['name', 'problem', 'scenario'],
                             columns='opset',
                             aggfunc=aggfunc).reset_index(drop=False)

    versions = ["opset%d" % t for t in range(1, piv.shape[1] - 2)]
    indices = ["name", "problem", "scenario"]
    piv.columns = indices + versions
    piv = piv[indices + list(reversed(versions))].copy()

    if "available-ERROR" in df.columns:
        piv2 = pandas.pivot_table(df, values="available-ERROR",
                                  index=['name', 'problem', 'scenario'],
                                  columns='opset',
                                  aggfunc=aggfunc).reset_index(drop=False)

        col = piv2.iloc[:, piv2.shape[1] - 1]
        piv["ERROR-msg"] = col
    return piv
