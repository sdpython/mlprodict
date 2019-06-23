"""
@file
@brief Runtime operator.
"""
import numpy


def _get_typed_class_attribute(self, k, atts):
    """
    Converts an attribute into a C++ value.
    """
    ty = atts[k]
    if isinstance(ty, numpy.ndarray):
        return getattr(self, k).astype(ty.dtype)
    elif isinstance(ty, bytes):
        return getattr(self, k).decode()
    elif isinstance(ty, list):
        return [_.decode() for _ in getattr(self, k)]
    elif isinstance(ty, int):
        return getattr(self, k)
    else:
        raise NotImplementedError("Unable to convert '{}' ({}).".format(
            k, getattr(self, k)))
