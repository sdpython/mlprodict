"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Tuple
import numpy as np
from .numpyx_core_api import cst, var, xapi
from .numpyx_types import (
    ElemType, OptParType, ParType, SequenceType, TensorType)


@xapi
def absolute(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return var(x, op='Abs')
