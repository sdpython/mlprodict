"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
from importlib import import_module
import numpy
import onnx
from sklearn.base import BaseEstimator
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version


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
    else:
        if len(arr.shape) == 1:
            threshold = 8
        else:
            threshold = min(
                50, min(50 // arr.shape[1], 8) * arr.shape[1])
        fLOG(numpy.array2string(arr, max_line_width=120,
                                suppress_small=True,
                                threshold=threshold))


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
