# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from collections import OrderedDict
import numpy
from ._op_helper import _get_typed_class_attribute
from ._op import OpRunClassifierProb
from .op_svm_classifier_ import RuntimeSVMClassifier  # pylint: disable=E0611


class SVMClassifier(OpRunClassifierProb):

    atts = OrderedDict([
        ('classlabels_int64s', numpy.empty(0, dtype=numpy.int64)),
        ('classlabels_strings', []),
        ('coefficients', numpy.empty(0, dtype=numpy.float32)),
        ('kernel_params', numpy.empty(0, dtype=numpy.float32)),
        ('kernel_type', b'NONE'),
        ('post_transform', b'NONE'),
        ('prob_a', numpy.empty(0, dtype=numpy.float32)),
        ('prob_b', numpy.empty(0, dtype=numpy.float32)),
        ('rho', numpy.empty(0, dtype=numpy.float32)),
        ('support_vectors', numpy.empty(0, dtype=numpy.float32)),
        ('vectors_per_class ', numpy.empty(0, dtype=numpy.float32)),
    ])

    def __init__(self, onnx_node, desc=None, **options):
        OpRunClassifierProb.__init__(self, onnx_node, desc=desc,
                                     expected_attributes=SVMClassifier.atts,
                                     **options)
        self._init()

    def _get_typed_attributes(self, k):
        return _get_typed_class_attribute(self, k, SVMClassifier.atts)

    def _init(self):
        self.rt_ = RuntimeSVMClassifier()
        atts = [self._get_typed_attributes(k)
                for k in SVMClassifier.atts]
        self.rt_.init(*atts)

    def _run(self, x):  # pylint: disable=W0221
        """
        This is a C++ implementation coming from
        :epkg:`onnxruntime`.
        `svm_regressor.cc
        <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_.
        See class :class:`RuntimeSVMClassifier
        <mlprodict.onnxrt.ops_cpu.op_svm_classifier_.RuntimeSVMClassifier>`.
        """
        label, scores = self.rt_.compute(x)
        if scores.shape[0] != label.shape[0]:
            scores = scores.reshape(label.shape[0],
                                    scores.shape[0] // label.shape[0])
        return (label, scores)
