"""
@file
@brief Importing this file takes time. It should be avoided.

.. versionadded:: 0.9
"""
import sys
from .xop_factory import _dynamic_class_creation


def _update_module():
    """
    Dynamically updates the module with operators defined by *ONNX*.
    """
    res = _dynamic_class_creation()
    this = sys.modules[__name__]
    unique = set()
    for cl in res:
        setattr(this, cl.__name__, cl)
        name = cl.__name__.split('_')[0]
        unique.add(name)
    res = _dynamic_class_creation(list(unique))
    for cl in res:
        setattr(this, cl.__name__, cl)


_update_module()
