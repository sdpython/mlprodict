# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ._new_ops import OperatorSchema


class FusedMatMul(OpRun):

    atts = {'alpha': 1., 'transA': 0, 'transB': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=FusedMatMul.atts,
                       **options)
        if self.transA:
            _meth = (FusedMatMul._fmatmul11 if self.transB
                     else FusedMatMul._fmatmul10)
        else:
            _meth = (FusedMatMul._fmatmul01 if self.transB
                     else FusedMatMul._fmatmul00)
        self._meth_ = _meth
        self._meth = lambda a, b: _meth(a, b, self.alpha)
        # more recent versions of the operator
        if not hasattr(self, "transBatchA"):
            self.transBatchA = 0
        if not hasattr(self, "transBatchB"):
            self.transBatchB = 0

    def _find_custom_operator_schema(self, op_name):
        if op_name == "FusedMatMul":
            return FusedMatMulSchema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")

    @staticmethod
    def _fmatmul00(a, b, alpha):
        return numpy.matmul(a, b) * alpha

    @staticmethod
    def _fmatmul01(a, b, alpha):
        return numpy.matmul(a, b.T) * alpha

    @staticmethod
    def _fmatmul10(a, b, alpha):
        return numpy.matmul(a.T, b) * alpha

    @staticmethod
    def _fmatmul11(a, b, alpha):
        return numpy.matmul(a.T, b.T) * alpha

    @staticmethod
    def _transpose(x, trans, transBatch):
        if trans:
            n = len(x.shape)
            perm = list(range(n-2)) + [n-2, n-1]
            x = numpy.transpose(x, perm)
        if transBatch:
            n = len(x.shape)
            perm = list(range(1, n-2)) + [0, n-1]
            x = numpy.transpose(x, perm)
        return x

    def _run(self, a, b, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.transBatchA or self.transBatchB or len(a.shape) != 2 or len(b.shape) != 2:
            ta = self._transpose(a, self.transA, self.transBatchA)
            tb = self._transpose(b, self.transB, self.transBatchB)
            try:
                return (numpy.matmul(ta, tb) * self.alpha, )
            except ValueError as e:
                raise ValueError(
                    f"Unable to multiply shape {a.shape}x{b.shape} "
                    f"({ta.shape}x{tb.shape}) "
                    f"with transA={self.transA}, "
                    f"transB={self.transB}, "
                    f"transBatchA={self.transBatchA}, "
                    f"transBatchB={self.transBatchB}, "
                    f"meth={self._meth_}.") from e
        try:
            return (self._meth(a, b), )
        except ValueError as e:
            raise ValueError(
                f"Unable to multiply shape {a.shape}x{b.shape} "
                f"with transA={self.transA}, "
                f"transB={self.transB}, "
                f"transBatchA={self.transBatchA}, "
                f"transBatchB={self.transBatchB}, "
                f"meth={self._meth_}.") from e


class FusedMatMulSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl FusedMatMul.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'FusedMatMul')
        self.attributes = FusedMatMul.atts
