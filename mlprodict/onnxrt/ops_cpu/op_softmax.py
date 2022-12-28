# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunUnaryNum, OpRunBinaryNum
from ._new_ops import OperatorSchema


class _Softmax(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=expected_attributes,
                               **options)

    def _run(self, X, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and X.flags['WRITEABLE']:
            return self._run_inplace(X)
        tmp = X - X.max(axis=self.axis, keepdims=1)
        Y = numpy.exp(tmp)
        Y /= Y.sum(axis=self.axis, keepdims=1)
        return (Y, )

    def _run_inplace(self, X):
        X -= X.max(axis=self.axis, keepdims=1)
        numpy.exp(X, out=X)
        X /= X.sum(axis=self.axis, keepdims=1)
        return (X, )

    def to_python(self, inputs):
        lines = ["tmp = {0} - {0}.max(axis=axis)[:, numpy.newaxis]".format(
            inputs[0]),
            "Y = numpy.exp(tmp)",
            "Y /= Y.sum(axis=axis)[:, numpy.newaxis]",
            "return Y"]
        return ("import numpy", "\n".join(lines))


class Softmax_1(_Softmax):

    atts = {'axis': 1}

    def __init__(self, onnx_node, desc=None, **options):
        _Softmax.__init__(self, onnx_node, desc=desc,
                          expected_attributes=Softmax_1.atts,
                          **options)


class Softmax_13(_Softmax):

    atts = {'axis': -1}

    def __init__(self, onnx_node, desc=None, **options):
        _Softmax.__init__(self, onnx_node, desc=desc,
                          expected_attributes=Softmax_13.atts,
                          **options)


class SoftmaxGrad_13(OpRunBinaryNum):
    """
    SoftmaxGrad computes :math:`dX = Y * ( dY - ReduceSum(Y * dY))`.
    ONNX does not have a dot product,
    which can be simulated as a pointwise-multiplication ("Mul"),
    followed by a "ReduceSum". Unfortunately, the treatment of "axis"
    is different in "SoftmaxGrad" and "ReduceSum".
    If axis=k for SoftmaxGrad, we need to specify [k, ..., n-1] as the axes of
    reduction for "ReduceSum", after accounting for negative-axis specification.
    An alternative solution would be to Flatten inputs to 2D and then reshape
    output back to original shape. Hopefully, many of these ops can be optimized
    away in the common-case of statically-known shapes.
    """

    atts = {'axis': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNum.__init__(self, onnx_node, desc=desc,
                                expected_attributes=SoftmaxGrad_13.atts,
                                **options)

    def _find_custom_operator_schema(self, op_name):
        if op_name in ("SoftmaxGrad_13", "SoftmaxGrad"):
            return SoftmaxGradSchema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")

    def _run(self, grad, prob, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        # softmax
        # tmp = X - X.max(axis=self.axis)[:, numpy.newaxis]
        # Y = numpy.exp(tmp)
        # Y /= Y.sum(axis=self.axis)[:, numpy.newaxis]
        # derivative
        pg = prob * grad
        if self.axis < 0:
            axis = len(pg.shape) + self.axis
        else:
            axis = self.axis
        axis = tuple(range(axis, len(pg.shape)))
        dg = grad - pg.sum(axis=axis, keepdims=1)
        return (prob * dg, )


class SoftmaxGradSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl SoftmaxGrad_13.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'SoftmaxGrad')
        self.attributes = SoftmaxGrad_13.atts


if onnx_opset_version() >= 13:
    Softmax = Softmax_13
else:  # pragma: no cover
    Softmax = Softmax_1

SoftmaxGrad = SoftmaxGrad_13
