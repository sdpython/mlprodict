"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from .numpyx_core_api import cst, var, xapi  # pylint: disable=W0611
from .numpyx_types import (  # pylint: disable=W0611
    ElemType, OptParType, ParType, SequenceType, TensorType)


@xapi
def absolute(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return var(x, op='Abs')


@xapi
def identity(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "Makes a copy."
    return var(x, op='Identity')
