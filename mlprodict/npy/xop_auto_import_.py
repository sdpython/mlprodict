"""
@file
@brief Xop API. Importing this file takes time. It should be avoided.

.. versionadded:: 0.9
"""
import sys
from .xop import _dynamic_class_creation


def _update_module():
    """
    Dynamically updates the module with operators defined by *ONNX*.
    """
    res = _dynamic_class_creation(include_past=True)
    this = sys.modules[__name__]
    unique = set()
    for cl in res:
        setattr(this, cl.__name__, cl)
        unique.add((cl.domain, cl.operator_name))
    res = _dynamic_class_creation(list(unique))
    for cl in res:
        setattr(this, cl.__name__, cl)


_update_module()
