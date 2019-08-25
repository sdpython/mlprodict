"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
from timeit import Timer
import os
import warnings
from importlib import import_module
import pickle
from time import perf_counter
import numpy
import onnx
from sklearn.base import BaseEstimator
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType, DataType
from ..conv.rewritten_converters import register_rewritten_operators


def modules_list():
    """
    Returns modules and versions currently used.

    .. runpython::
        :showcode:
        :rst:

        from mlprodict.onnxrt.validate.validate_helper import modules_list
        from pyquickhelper.pandashelper import df2rst
        from pandas import DataFrame
        print(df2rst(DataFrame(modules_list())))
    """
    def try_import(name):
        try:
            mod = import_module(name)
        except ImportError:
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
    else:
        s = str(arr)
        if len(s) > 50:
            s = s[:50] + "..."
        fLOG(s)


def get_opset_number_from_onnx():
    """
    Retuns the current :epkg:`onnx` opset
    based on the installed version of :epkg:`onnx`.
    """
    return onnx.defs.onnx_opset_version()


def sklearn_operators(subfolder=None, extended=False):
    """
    Builds the list of operators from :epkg:`scikit-learn`.
    The function goes through the list of submodule
    and get the list of class which inherit from
    :epkg:`scikit-learn:base:BaseEstimator`.

    @param      subfolder   look into only one subfolder
    @param      extended    extends the list to the list of operators
                            this package implements a converter for
    """
    subfolders = sklearn__all__ + ['mlprodict.onnx_conv']
    found = []
    for subm in sorted(subfolders):
        if isinstance(subm, list):
            continue
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
                                   'BaseEnsemble'}:
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
            name = m.__module__.split('.')
            sub = '.'.join(name[1:])
            pack = name[0]
            if m.__name__ not in done:
                found.append(
                    dict(name=m.__name__, cl=m, package=pack, sub=sub))
    return found


def to_onnx(model, X=None, name=None, initial_types=None,
            target_opset=None, options=None,
            dtype=numpy.float32, rewrite_ops=False):
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
    @param      options         additional parameters for the conversion
    @param      dtype           type to use to convert the model
    @param      rewrite_ops     rewrites some existing converters,
                                the changes are permanent
    @return                     converted model

    The function rewrites function *to_onnx* from :epkg:`sklearn-onnx`
    but may changes a few converters if *rewrite_ops* is True.
    For example, :epkg:`ONNX` only supports *TreeEnsembleRegressor*
    for float but not for double. It becomes available
    if ``dtype=numpy.float64`` and ``rewrite_ops=True``.
    """
    from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
    from skl2onnx.algebra.type_helper import guess_initial_types
    from skl2onnx import convert_sklearn

    if isinstance(model, OnnxOperatorMixin):
        if target_opset is not None:
            raise NotImplementedError(
                "target_opset not yet implemented for OnnxOperatorMixin.")
        if options is not None:
            raise NotImplementedError(
                "options not yet implemented for OnnxOperatorMixin.")
        return model.to_onnx(X=X, name=name, dtype=dtype)
    if name is None:
        name = "ONNX(%s)" % model.__class__.__name__
    initial_types = guess_initial_types(X, initial_types)
    if dtype is None:
        raise RuntimeError("dtype cannot be None")
    if isinstance(dtype, FloatTensorType):
        dtype = numpy.float32
    elif isinstance(dtype, DoubleTensorType):
        dtype = numpy.float64
    new_dtype = dtype
    if isinstance(dtype, numpy.ndarray):
        new_dtype = dtype.dtype
    elif isinstance(dtype, DataType):
        new_dtype = numpy.float32
    if new_dtype not in (numpy.float32, numpy.float64, numpy.int64,
                         numpy.int32):
        raise NotImplementedError(
            "dtype should be real not {} ({})".format(new_dtype, dtype))
    if rewrite_ops:
        old_values = register_rewritten_operators()
    else:
        old_values = None
    try:
        res = convert_sklearn(model, initial_types=initial_types, name=name,
                              target_opset=target_opset, options=options,
                              dtype=new_dtype)
    except TypeError:
        # older version of sklearn-onnx
        res = convert_sklearn(model, initial_types=initial_types, name=name,
                              target_opset=target_opset, options=options)
    if old_values is not None:
        register_rewritten_operators(old_values)
    return res


def _measure_time(fct, repeat=1, number=1):
    """
    Measures the execution time for a function.

    @param      fct     function to measure
    @param      repeat  number of times to repeat
    @param      number  number of times between two measures
    @return             last result, average, values
    """
    res = None
    values = []
    for __ in range(repeat):
        begin = perf_counter()
        for _ in range(number):
            res = fct()
        end = perf_counter()
        values.append(end - begin)
    if repeat * number == 1:
        return res, values[0], values
    else:
        return res, sum(values) / (repeat * number), values


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

    @param      dump_folder     dump_folder
    @param      obs_op          obs_op (information)
    @param      is_error        is it an error or not?
    @kwargs                     kwargs
    """
    parts = (obs_op['runtime'], obs_op['name'], obs_op['scenario'],
             obs_op['problem'], obs_op.get('opset', '-'))
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


def default_time_kwargs():
    """
    Returns default values *number* and *repeat* to measure
    the execution of a function.

    .. runpython::
        :showcode:

        from mlprodict.onnxrt.validate.validate_helper import default_time_kwargs
        import pprint
        pprint.pprint(default_time_kwargs())

    keys define the number of rows,
    values defines *number* and *repeat*.
    """
    return {
        1: dict(number=30, repeat=20),
        10: dict(number=20, repeat=20),
        100: dict(number=8, repeat=5),
        1000: dict(number=5, repeat=5),
        10000: dict(number=3, repeat=3),
        100000: dict(number=1, repeat=1),
    }


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
