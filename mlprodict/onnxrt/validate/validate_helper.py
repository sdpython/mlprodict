"""
@file
@brief Validates runtime for many :epkg:`scikit-learn` operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
import math
import copy
from timeit import Timer
import os
import warnings
from importlib import import_module
import pickle
from time import perf_counter
import numpy
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearModel
from sklearn.model_selection import train_test_split
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from .validate_problems import _problems


class RuntimeBadResultsError(RuntimeError):
    """
    Raised when the results are too different from
    :epkg:`scikit-learn`.
    """

    def __init__(self, msg, obs):
        """
        :param msg: to display
        :param obs: observations
        """
        RuntimeError.__init__(self, msg)
        self.obs = obs


def _dictionary2str(di):
    el = []
    for k in sorted(di):
        el.append('{}={}'.format(k, di[k]))
    return '/'.join(el)


def modules_list():
    """
    Returns modules and versions currently used.

    .. runpython::
        :showcode:
        :rst:
        :warningout: DeprecationWarning

        from mlprodict.onnxrt.validate.validate_helper import modules_list
        from pyquickhelper.pandashelper import df2rst
        from pandas import DataFrame
        print(df2rst(DataFrame(modules_list())))
    """
    def try_import(name):
        try:
            mod = import_module(name)
        except ImportError:  # pragma: no cover
            return None
        return (dict(name=name, version=mod.__version__)
                if hasattr(mod, '__version__') else dict(name=name))

    rows = []
    for name in sorted(['pandas', 'numpy', 'sklearn', 'mlprodict',
                        'skl2onnx', 'onnxmltools', 'onnx', 'onnxruntime',
                        'scipy']):
        res = try_import(name)
        if res is not None:
            rows.append(res)
    return rows


def _dispsimple(arr, fLOG):
    if isinstance(arr, (tuple, list)):
        for i, a in enumerate(arr):
            fLOG("output %d" % i)
            _dispsimple(a, fLOG)
    elif hasattr(arr, 'shape'):
        if len(arr.shape) == 1:
            threshold = 8
        else:
            threshold = min(
                50, min(50 // arr.shape[1], 8) * arr.shape[1])
        fLOG(numpy.array2string(arr, max_line_width=120,
                                suppress_small=True,
                                threshold=threshold))
    else:  # pragma: no cover
        s = str(arr)
        if len(s) > 50:
            s = s[:50] + "..."
        fLOG(s)


def _merge_options(all_conv_options, aoptions):
    if aoptions is None:
        return copy.deepcopy(all_conv_options)
    if not isinstance(aoptions, dict):
        return copy.deepcopy(aoptions)  # pragma: no cover
    merged = {}
    for k, v in all_conv_options.items():
        if k in aoptions:
            merged[k] = _merge_options(v, aoptions[k])
        else:
            merged[k] = copy.deepcopy(v)
    for k, v in aoptions.items():
        if k in all_conv_options:
            continue
        merged[k] = copy.deepcopy(v)
    return merged


def sklearn_operators(subfolder=None, extended=False,
                      experimental=True):
    """
    Builds the list of operators from :epkg:`scikit-learn`.
    The function goes through the list of submodule
    and get the list of class which inherit from
    :epkg:`scikit-learn:base:BaseEstimator`.

    :param subfolder: look into only one subfolder
    :param extended: extends the list to the list of operators
        this package implements a converter for
    :param experimental: includes experimental module from
        :epkg:`scikit-learn` (see `sklearn.experimental
        <https://github.com/scikit-learn/scikit-learn/
        tree/master/sklearn/experimental>`_)
    :return: the list of found operators
    """
    if experimental:
        from sklearn.experimental import (  # pylint: disable=W0611
            enable_hist_gradient_boosting,
            enable_iterative_imputer)

    subfolders = sklearn__all__ + ['mlprodict.onnx_conv']
    found = []
    for subm in sorted(subfolders):
        if isinstance(subm, list):
            continue  # pragma: no cover
        if subfolder is not None and subm != subfolder:
            continue

        if subm == 'feature_extraction':
            subs = [subm, 'feature_extraction.text']
        else:
            subs = [subm]

        for sub in subs:
            if '.' in sub and sub not in {'feature_extraction.text'}:
                name_sub = sub
            else:
                name_sub = "{0}.{1}".format("sklearn", sub)
            try:
                mod = import_module(name_sub)
            except ModuleNotFoundError:
                continue

            if hasattr(mod, "register_converters"):
                fct = getattr(mod, "register_converters")
                cls = fct()
            else:
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
                                   'FeatureUnion', 'BaseEstimator',
                                   'BaseEnsemble', 'BaseDecisionTree'}:
                    continue
                if cl.__name__ in {'CustomScorerTransform'}:
                    continue
                if (sub in {'calibration', 'dummy', 'manifold'} and
                        'Calibrated' not in cl.__name__):
                    continue
                if issub:
                    pack = "sklearn" if sub in sklearn__all__ else cl.__module__.split('.')[
                        0]
                    found.append(
                        dict(name=cl.__name__, subfolder=sub, cl=cl, package=pack))

    if extended:
        from ...onnx_conv import register_converters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            models = register_converters(True)

        done = set(_['name'] for _ in found)
        for m in models:
            try:
                name = m.__module__.split('.')
            except AttributeError as e:  # pragma: no cover
                raise AttributeError("Unexpected value, m={}".format(m)) from e
            sub = '.'.join(name[1:])
            pack = name[0]
            if m.__name__ not in done:
                found.append(
                    dict(name=m.__name__, cl=m, package=pack, sub=sub))

    # let's remove models which cannot predict
    all_found = found
    found = []
    for mod in all_found:
        cl = mod['cl']
        if hasattr(cl, 'fit_predict') and not hasattr(cl, 'predict'):
            continue
        if hasattr(cl, 'fit_transform') and not hasattr(cl, 'transform'):
            continue
        if (not hasattr(cl, 'transform') and
                not hasattr(cl, 'predict') and
                not hasattr(cl, 'decision_function')):
            continue
        found.append(mod)
    return found


def _measure_time(fct, repeat=1, number=1, first_run=True):
    """
    Measures the execution time for a function.

    :param fct: function to measure
    :param repeat: number of times to repeat
    :param number: number of times between two measures
    :param first_run: if True, runs the function once before measuring
    :return: last result, average, values
    """
    res = None
    values = []
    if first_run:
        fct()
    for __ in range(repeat):
        begin = perf_counter()
        for _ in range(number):
            res = fct()
        end = perf_counter()
        values.append(end - begin)
    if repeat * number == 1:
        return res, values[0], values
    return res, sum(values) / (repeat * number), values  # pragma: no cover


def _shape_exc(obj):
    if hasattr(obj, 'shape'):
        return obj.shape
    if isinstance(obj, (list, dict, tuple)):
        return "[{%d}]" % len(obj)
    return None


def dump_into_folder(dump_folder, obs_op=None, is_error=True,
                     **kwargs):
    """
    Dumps information when an error was detected
    using :epkg:`*py:pickle`.

   :param dump_folder: dump_folder
   :param obs_op: obs_op (information)
   :param is_error: is it an error or not?
   :param kwargs: additional parameters
    :return: name
    """
    if dump_folder is None:
        raise ValueError("dump_folder cannot be None.")
    optim = obs_op.get('optim', '')
    optim = str(optim)
    optim = optim.replace("<class 'sklearn.", "")
    optim = optim.replace("<class '", "")
    optim = optim.replace(" ", "")
    optim = optim.replace(">", "")
    optim = optim.replace("=", "")
    optim = optim.replace("{", "")
    optim = optim.replace("}", "")
    optim = optim.replace(":", "")
    optim = optim.replace("'", "")
    optim = optim.replace("/", "")
    optim = optim.replace("\\", "")
    parts = (obs_op['runtime'], obs_op['name'], obs_op['scenario'],
             obs_op['problem'], optim,
             "op" + str(obs_op.get('opset', '-')),
             "nf" + str(obs_op.get('n_features', '-')))
    name = "dump-{}-{}.pkl".format(
        "ERROR" if is_error else "i",
        "-".join(map(str, parts)))
    name = os.path.join(dump_folder, name)
    obs_op = obs_op.copy()
    fcts = [k for k in obs_op if k.startswith('lambda')]
    for fct in fcts:
        del obs_op[fct]
    kwargs.update({'obs_op': obs_op})
    with open(name, "wb") as f:
        pickle.dump(kwargs, f)
    return name


def default_time_kwargs():
    """
    Returns default values *number* and *repeat* to measure
    the execution of a function.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.onnxrt.validate.validate_helper import default_time_kwargs
        import pprint
        pprint.pprint(default_time_kwargs())

    keys define the number of rows,
    values defines *number* and *repeat*.
    """
    return {
        1: dict(number=30, repeat=20),
        10: dict(number=20, repeat=20),
        100: dict(number=8, repeat=10),
        1000: dict(number=5, repeat=5),
        10000: dict(number=3, repeat=3),
    }


def measure_time(stmt, x, repeat=10, number=50, div_by_number=False, first_run=True):
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string
    :param x: matrix
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param div_by_number: divide by the number of executions
    :param first_run: if True, runs the function once before measuring
    :return: dictionary

    See `Timer.repeat <https://docs.python.org/3/library/timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    if x is None:
        raise ValueError("x cannot be None")  # pragma: no cover

    try:
        stmt(x)
    except RuntimeError as e:  # pragma: no cover
        raise RuntimeError("{}-{}".format(type(x), x.dtype)) from e

    def fct():
        stmt(x)

    if first_run:
        fct()
    tim = Timer(fct)
    res = numpy.array(tim.repeat(repeat=repeat, number=number))
    total = numpy.sum(res)
    if div_by_number:
        res /= number
    mean = numpy.mean(res)
    dev = numpy.mean(res ** 2)
    dev = max(0, (dev - mean**2)) ** 0.5
    mes = dict(average=mean, deviation=dev, min_exec=numpy.min(res),
               max_exec=numpy.max(res), repeat=repeat, number=number,
               total=total)
    return mes


def _multiply_time_kwargs(time_kwargs, time_kwargs_fact, inst):
    """
    Multiplies values in *time_kwargs* following strategy
    *time_kwargs_fact* for a given model *inst*.

    :param time_kwargs: see below
    :param time_kwargs_fact: see below
    :param inst: :epkg:`scikit-learn` model
    :return: new *time_kwargs*

    Possible values for *time_kwargs_fact*:

    - a integer: multiplies *number* by this number
    - `'lin'`: multiplies value *number* for linear models depending
      on the number of rows to process (:math:`\\propto 1/\\log_{10}(n)`)

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from pprint import pprint
        from sklearn.linear_model import LinearRegression
        from mlprodict.onnxrt.validate.validate_helper import (
            default_time_kwargs, _multiply_time_kwargs)

        lr = LinearRegression()
        kw = default_time_kwargs()
        pprint(kw)

        kw2 = _multiply_time_kwargs(kw, 'lin', lr)
        pprint(kw2)
    """
    if time_kwargs is None:
        raise ValueError("time_kwargs cannot be None.")  # pragma: no cover
    if time_kwargs_fact in ('', None):
        return time_kwargs
    try:
        vi = int(time_kwargs_fact)
        time_kwargs_fact = vi
    except (TypeError, ValueError):
        pass
    if isinstance(time_kwargs_fact, int):
        time_kwargs_modified = copy.deepcopy(time_kwargs)
        for k in time_kwargs_modified:
            time_kwargs_modified[k]['number'] *= time_kwargs_fact
        return time_kwargs_modified
    if time_kwargs_fact == 'lin':
        if isinstance(inst, LinearModel):
            time_kwargs_modified = copy.deepcopy(time_kwargs)
            for k in time_kwargs_modified:
                kl = max(int(math.log(k) / math.log(10) + 1e-5), 1)
                f = max(int(10 / kl + 0.5), 1)
                time_kwargs_modified[k]['number'] *= f
                time_kwargs_modified[k]['repeat'] *= 1
            return time_kwargs_modified
        return time_kwargs
    raise ValueError(  # pragma: no cover
        "Unable to interpret time_kwargs_fact='{}'.".format(
            time_kwargs_fact))


def _get_problem_data(prob, n_features):
    data_problem = _problems[prob](n_features=n_features)
    if len(data_problem) == 6:
        X_, y_, init_types, method, output_index, Xort_ = data_problem
        dofit = True
    elif len(data_problem) == 7:
        X_, y_, init_types, method, output_index, Xort_, dofit = data_problem
    else:
        raise RuntimeError(  # pragma: no cover
            "Unable to interpret problem '{}'.".format(prob))
    if (len(X_.shape) == 2 and X_.shape[1] != n_features and
            n_features is not None):
        raise RuntimeError(  # pragma: no cover
            "Problem '{}' with n_features={} returned {} features"
            "(func={}).".format(prob, n_features, X_.shape[1],
                                _problems[prob]))
    if y_ is None:
        (X_train, X_test, Xort_train,  # pylint: disable=W0612
            Xort_test) = train_test_split(
                X_, Xort_, random_state=42)
        y_train, y_test = None, None
    else:
        (X_train, X_test, y_train, y_test,  # pylint: disable=W0612
            Xort_train, Xort_test) = train_test_split(
                X_, y_, Xort_, random_state=42)
    if isinstance(init_types, tuple):
        init_types, conv_options = init_types
    else:
        conv_options = None

    if isinstance(method, tuple):
        method_name, predict_kwargs = method
    else:
        method_name = method
        predict_kwargs = {}

    return (X_train, X_test, y_train,
            y_test, Xort_test,
            init_types, conv_options, method_name,
            output_index, dofit, predict_kwargs)
