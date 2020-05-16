"""
@file
@brief Inspired from skl2onnx, handles two backends.
"""
import numpy
from ...tools.asv_options_helper import get_opset_number_from_onnx
from .utils_backend_onnxruntime import _capture_output


from .tests_helper import dump_data_and_model  # noqa
from .tests_helper import (  # noqa
    dump_one_class_classification,
    dump_binary_classification,
    dump_multilabel_classification,
    dump_multiple_classification,
)
from .tests_helper import (  # noqa
    dump_multiple_regression,
    dump_single_regression,
    convert_model,
    fit_classification_model,
    fit_classification_model_simple,
    fit_multilabel_classification_model,
    fit_regression_model,
)


def create_tensor(N, C, H=None, W=None):
    "Creates a tensor."
    if H is None and W is None:
        return numpy.random.rand(N, C).astype(numpy.float32, copy=False)  # pylint: disable=E1101
    elif H is not None and W is not None:
        return numpy.random.rand(N, C, H, W).astype(numpy.float32, copy=False)  # pylint: disable=E1101
    raise ValueError('This function only produce 2-D or 4-D tensor.')


def _get_ir_version(opv):
    if opv >= 12:
        return 7
    if opv >= 11:
        return 6
    if opv >= 10:
        return 5
    if opv >= 9:
        return 4
    if opv >= 8:
        return 4
    return 3


TARGET_OPSET = get_opset_number_from_onnx()
TARGET_IR = _get_ir_version(TARGET_OPSET)
