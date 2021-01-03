# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from collections import OrderedDict
import numpy
from ._op_helper import _get_typed_class_attribute
from ._op import OpRunClassifierProb, RuntimeTypeError
from ._op_classifier_string import _ClassifierCommon
from ._new_ops import OperatorSchema
from .op_svm_classifier_ import (  # pylint: disable=E0611,E0401
    RuntimeSVMClassifierFloat,
    RuntimeSVMClassifierDouble,
)


class SVMClassifierCommon(OpRunClassifierProb, _ClassifierCommon):

    def __init__(self, dtype, onnx_node, desc=None,
                 expected_attributes=None, **options):
        OpRunClassifierProb.__init__(self, onnx_node, desc=desc,
                                     expected_attributes=expected_attributes,
                                     **options)
        self._init(dtype=dtype)

    def _get_typed_attributes(self, k):
        return _get_typed_class_attribute(self, k, self.__class__.atts)

    def _find_custom_operator_schema(self, op_name):
        """
        Finds a custom operator defined by this runtime.
        """
        if op_name == "SVMClassifierDouble":
            return SVMClassifierDoubleSchema()
        raise RuntimeError(  # pragma: no cover
            "Unable to find a schema for operator '{}'.".format(op_name))

    def _init(self, dtype):
        self._post_process_label_attributes()
        if dtype == numpy.float32:
            self.rt_ = RuntimeSVMClassifierFloat(20)
        elif dtype == numpy.float64:
            self.rt_ = RuntimeSVMClassifierDouble(20)
        else:
            raise RuntimeTypeError(  # pragma: no cover
                "Unsupported dtype={}.".format(dtype))
        atts = [self._get_typed_attributes(k)
                for k in SVMClassifier.atts]
        self.rt_.init(*atts)

    def _run(self, x):  # pylint: disable=W0221
        """
        This is a C++ implementation coming from
        :epkg:`onnxruntime`.
        `svm_classifier.cc
        <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc>`_.
        See class :class:`RuntimeSVMClassifier
        <mlprodict.onnxrt.ops_cpu.op_svm_classifier_.RuntimeSVMClassifier>`.
        """
        label, scores = self.rt_.compute(x)
        if scores.shape[0] != label.shape[0]:
            scores = scores.reshape(label.shape[0],
                                    scores.shape[0] // label.shape[0])
        return self._post_process_predicted_label(label, scores)


class SVMClassifier(SVMClassifierCommon):

    atts = OrderedDict([
        ('classlabels_ints', numpy.empty(0, dtype=numpy.int64)),
        ('classlabels_strings', []),
        ('coefficients', numpy.empty(0, dtype=numpy.float32)),
        ('kernel_params', numpy.empty(0, dtype=numpy.float32)),
        ('kernel_type', b'NONE'),
        ('post_transform', b'NONE'),
        ('prob_a', numpy.empty(0, dtype=numpy.float32)),
        ('prob_b', numpy.empty(0, dtype=numpy.float32)),
        ('rho', numpy.empty(0, dtype=numpy.float32)),
        ('support_vectors', numpy.empty(0, dtype=numpy.float32)),
        ('vectors_per_class', numpy.empty(0, dtype=numpy.float32)),
    ])

    def __init__(self, onnx_node, desc=None, **options):
        SVMClassifierCommon.__init__(
            self, numpy.float32, onnx_node, desc=desc,
            expected_attributes=SVMClassifier.atts,
            **options)


class SVMClassifierDouble(SVMClassifierCommon):

    atts = OrderedDict([
        ('classlabels_ints', numpy.empty(0, dtype=numpy.int64)),
        ('classlabels_strings', []),
        ('coefficients', numpy.empty(0, dtype=numpy.float64)),
        ('kernel_params', numpy.empty(0, dtype=numpy.float64)),
        ('kernel_type', b'NONE'),
        ('post_transform', b'NONE'),
        ('prob_a', numpy.empty(0, dtype=numpy.float64)),
        ('prob_b', numpy.empty(0, dtype=numpy.float64)),
        ('rho', numpy.empty(0, dtype=numpy.float64)),
        ('support_vectors', numpy.empty(0, dtype=numpy.float64)),
        ('vectors_per_class', numpy.empty(0, dtype=numpy.float64)),
    ])

    def __init__(self, onnx_node, desc=None, **options):
        SVMClassifierCommon.__init__(
            self, numpy.float64, onnx_node, desc=desc,
            expected_attributes=SVMClassifierDouble.atts,
            **options)


class SVMClassifierDoubleSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl SVMClassifierDouble.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'SVMClassifierDouble')
        self.attributes = SVMClassifierDouble.atts
