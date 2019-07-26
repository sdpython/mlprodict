"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
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


def modules_list():
    """
    Returns modules and versions currently used.

    .. runpython::
        :showcode:
        :rst:

        from mlprodict.onnxrt.validate_helper import modules_list
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
    found = []
    for subm in sorted(sklearn__all__):
        if isinstance(subm, list):
            continue
        if subfolder is not None and subm != subfolder:
            continue

        if subm == 'feature_extraction':
            subs = [subm, 'feature_extraction.text']
        else:
            subs = [subm]

        for sub in subs:
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
                                   'FeatureUnion', 'BaseEstimator',
                                   'BaseEnsemble'}:
                    continue
                if (sub in {'calibration', 'dummy', 'manifold'} and
                        'Calibrated' not in cl.__name__):
                    continue
                if issub:
                    found.append(
                        dict(name=cl.__name__, subfolder=sub, cl=cl, package='sklearn'))

    if extended:
        from ..onnx_conv import register_converters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            models = register_converters(True)
        for m in models:
            name = m.__module__.split('.')
            sub = '.'.join(name[1:])
            pack = name[0]
            found.append(dict(name=m.__name__, cl=m, package=pack, sub=sub))
    return found


def to_onnx(model, X=None, name=None, initial_types=None,
            target_opset=None, options=None, dtype=None):
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
    @return                     converted model
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
    return convert_sklearn(model, initial_types=initial_types, name=name,
                           target_opset=target_opset, options=options,
                           dtype=new_dtype)


def _measure_time(fct):
    """
    Measures the execution time for a function.
    """
    begin = perf_counter()
    res = fct()
    end = perf_counter()
    return res, end - begin


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
