"""
@file
@brief Inspired from sklearn-onnx, handles two backends.
"""
import numpy
from .utils_backend_onnxruntime import _capture_output


from .tests_helper import (  # noqa
    binary_array_to_string,
    dump_data_and_model,
    dump_one_class_classification,
    dump_binary_classification,
    dump_multilabel_classification,
    dump_multiple_classification,
    dump_multiple_regression,
    dump_single_regression,
    convert_model,
    fit_classification_model,
    fit_classification_model_simple,
    fit_multilabel_classification_model,
    fit_regression_model)


def create_tensor(N, C, H=None, W=None):
    "Creates a tensor."
    if H is None and W is None:
        return numpy.random.rand(N, C).astype(numpy.float32, copy=False)  # pylint: disable=E1101
    elif H is not None and W is not None:
        return numpy.random.rand(N, C, H, W).astype(numpy.float32, copy=False)  # pylint: disable=E1101
    raise ValueError(  # pragma no cover
        'This function only produce 2-D or 4-D tensor.')


def ort_version_greater(ver):
    """
    Tells if onnxruntime version is greater than *ver*.

    :param ver: version as a string
    :return: boolean
    """
    from onnxruntime import __version__
    from pyquickhelper.texthelper.version_helper import compare_module_version
    return compare_module_version(__version__, ver) >= 0
