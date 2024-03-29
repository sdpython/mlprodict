"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""

DEFAULT_OPSETS = {'': 18, 'ai.onnx.ml': 3}
FUNCTION_DOMAIN = "FUNCTION-DOMAIN"
ONNX_DOMAIN = "ONNX-DOMAIN"

_OPSET_TO_IR_VERSION = {
    14: 7,
    15: 8,
    16: 8,
    17: 8,
    18: 8,
    19: 9,
}

DEFAULT_IR_VERSION = _OPSET_TO_IR_VERSION[DEFAULT_OPSETS[""]]
