# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ._new_ops import OperatorSchema
from .op_murmurhash3_ import (  # pragma: disable=E0611
    MurmurHash3_x86_32, MurmurHash3_x86_32_positive)


class MurmurHash3(OpRun):

    atts = {'positive': 1, 'seed': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=MurmurHash3.atts,
                       **options)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "MurmurHash3":
            return MurmurHash3Schema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.positive:
            res = numpy.empty(x.shape, dtype=numpy.uint32).flatten()
            xf = x.flatten()
            for i in range(len(xf)):  # pylint: disable=C0200
                res[i] = MurmurHash3_x86_32_positive(xf[i], self.seed)
            return (res.reshape(x.shape), )

        res = numpy.empty(x.shape, dtype=numpy.int32).flatten()
        xf = x.flatten()
        for i in range(len(xf)):  # pylint: disable=C0200
            res[i] = MurmurHash3_x86_32(xf[i], self.seed)
        return (res.reshape(x.shape), )


class MurmurHash3Schema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl MurmurHash3.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'MurmurHash3')
        self.attributes = MurmurHash3.atts
