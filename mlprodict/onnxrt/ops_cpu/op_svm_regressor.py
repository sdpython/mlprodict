# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from collections import OrderedDict
import numpy
from ._op_helper import _get_typed_class_attribute
from ._op import OpRun
from .op_svm_regressor_ import RuntimeSVMRegressor  # pylint: disable=E0611


class SVMRegressor(OpRun):

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
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=SVMRegressor.atts,
                       **options)
        self._init()

    def _get_typed_attributes(self, k):
        return _get_typed_class_attribute(self, k, SVMRegressor.atts)

    def _init(self):
        self.rt_ = RuntimeSVMRegressor()
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
