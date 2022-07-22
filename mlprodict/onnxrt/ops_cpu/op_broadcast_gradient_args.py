# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ._new_ops import OperatorSchema


class BroadcastGradientArgs(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "BroadcastGradientArgs":
            return BroadcastGradientArgsSchema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")

    def _run(self, a_shape, b_shape, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221

        A_dims = a_shape
        B_dims = b_shape
        a_size = len(a_shape)
        b_size = len(b_shape)

        ndim = max(a_size, b_size)

        i = a_size - 1
        j = b_size - 1
        k = ndim - 1

        a_axes = []
        b_axes = []

        while i >= 0 and j >= 0:
            A_dim = A_dims[i]
            B_dim = B_dims[j]

            if A_dim != B_dim:
                if A_dim == 1:
                    a_axes.append(k)
                elif B_dim == 1:
                    b_axes.append(k)
                else:
                    a = A_dims[:a_size]
                    b = B_dims[:b_size]
                    raise RuntimeError(
                        "Broadcast is not possible between inputs of "
                        "shapes: %r and %r." % (a, b))
            i -= 1
            j -= 1
            k -= 1

        if i < 0:
            while k >= 0:
                a_axes.append(k)
                k -= 1
        else:
            while k >= 0:
                b_axes.append(k)
                k -= 1

        return (numpy.array(a_axes, dtype=numpy.int64),
                numpy.array(b_axes, dtype=numpy.int64))


class BroadcastGradientArgsSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl BroadcastGradientArgs.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'BroadcastGradientArgs')
        self.attributes = BroadcastGradientArgs.atts
