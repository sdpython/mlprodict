# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from collections import OrderedDict
import numpy
from ._op_helper import _get_typed_class_attribute
from ._op import OpRunUnaryNum, RuntimeTypeError
from ._new_ops import OperatorSchema
from .op_svm_regressor_ import (  # pylint: disable=E0611,E0401
    RuntimeSVMRegressorFloat,
    RuntimeSVMRegressorDouble,
)


class SVMRegressorCommon(OpRunUnaryNum):

    def __init__(self, dtype, onnx_node, desc=None,
                 expected_attributes=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=expected_attributes,
                               **options)
        self._init(dtype=dtype)

    def _get_typed_attributes(self, k):
        return _get_typed_class_attribute(self, k, self.__class__.atts)

    def _find_custom_operator_schema(self, op_name):
        """
        Finds a custom operator defined by this runtime.
        """
        if op_name == "SVMRegressorDouble":
            return SVMRegressorDoubleSchema()
        raise RuntimeError(  # pragma: no cover
            "Unable to find a schema for operator '{}'.".format(op_name))

    def _init(self, dtype):
        if dtype == numpy.float32:
            self.rt_ = RuntimeSVMRegressorFloat(50)
        elif dtype == numpy.float64:
            self.rt_ = RuntimeSVMRegressorDouble(50)
        else:
            raise RuntimeTypeError(  # pragma: no cover
                "Unsupported dtype={}.".format(dtype))
        atts = [self._get_typed_attributes(k)
                for k in SVMRegressor.atts]
        self.rt_.init(*atts)

    def _run(self, x):  # pylint: disable=W0221
        """
        This is a C++ implementation coming from
        :epkg:`onnxruntime`.
        `svm_regressor.cc
        <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_.
        See class :class:`RuntimeSVMRegressor
        <mlprodict.onnxrt.ops_cpu.op_svm_regressor_.RuntimeSVMRegressor>`.
        """
        pred = self.rt_.compute(x)
        if pred.shape[0] != x.shape[0]:
            pred = pred.reshape(x.shape[0], pred.shape[0] // x.shape[0])
        return (pred, )


class SVMRegressor(SVMRegressorCommon):

    atts = OrderedDict([
        ('coefficients', numpy.empty(0, dtype=numpy.float32)),
        ('kernel_params', numpy.empty(0, dtype=numpy.float32)),
        ('kernel_type', b'NONE'),
        ('n_supports', 0),
        ('one_class', 0),
        ('post_transform', b'NONE'),
        ('rho', numpy.empty(0, dtype=numpy.float32)),
        ('support_vectors', numpy.empty(0, dtype=numpy.float32)),
    ])

    def __init__(self, onnx_node, desc=None, **options):
        SVMRegressorCommon.__init__(
            self, numpy.float32, onnx_node, desc=desc,
            expected_attributes=SVMRegressor.atts,
            **options)


class SVMRegressorDouble(SVMRegressorCommon):

    atts = OrderedDict([
        ('coefficients', numpy.empty(0, dtype=numpy.float64)),
        ('kernel_params', numpy.empty(0, dtype=numpy.float64)),
        ('kernel_type', b'NONE'),
        ('n_supports', 0),
        ('one_class', 0),
        ('post_transform', b'NONE'),
        ('rho', numpy.empty(0, dtype=numpy.float64)),
        ('support_vectors', numpy.empty(0, dtype=numpy.float64)),
    ])

    def __init__(self, onnx_node, desc=None, **options):
        SVMRegressorCommon.__init__(
            self, numpy.float64, onnx_node, desc=desc,
            expected_attributes=SVMRegressorDouble.atts,
            **options)


class SVMRegressorDoubleSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl SVMRegressorDouble.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'SVMRegressorDouble')
        self.attributes = SVMRegressorDouble.atts
