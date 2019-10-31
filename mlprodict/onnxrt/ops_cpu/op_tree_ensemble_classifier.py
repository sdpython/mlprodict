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
from ._new_ops import OperatorSchema
from .op_tree_ensemble_classifier_ import (  # pylint: disable=E0611
    RuntimeTreeEnsembleClassifierDouble,
    RuntimeTreeEnsembleClassifierFloat,
)


class TreeEnsembleClassifierCommon(OpRunClassifierProb):

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
        if op_name == "TreeEnsembleClassifierDouble":
            return TreeEnsembleClassifierDoubleSchema()
        raise RuntimeError(
            "Unable to find a schema for operator '{}'.".format(op_name))

    def _init(self, dtype):
        if dtype == numpy.float32:
            self.rt_ = RuntimeTreeEnsembleClassifierFloat()
        elif dtype == numpy.float64:
            self.rt_ = RuntimeTreeEnsembleClassifierDouble()
        else:
            raise RuntimeTypeError("Unsupported dtype={}.".format(dtype))
        atts = [self._get_typed_attributes(k)
                for k in self.__class__.atts]
        self.rt_.init(*atts)

    def _run(self, x):  # pylint: disable=W0221
        """
        This is a C++ implementation coming from
        :epkg:`onnxruntime`.
        `tree_ensemble_classifier.cc
        <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc>`_.
        See class :class:`RuntimeTreeEnsembleClassifier
        <mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_.RuntimeTreeEnsembleClassifier>`.
        """
        label, scores = self.rt_.compute(x)
        if scores.shape[0] != label.shape[0]:
            scores = scores.reshape(label.shape[0],
                                    scores.shape[0] // label.shape[0])
        return (label, scores)


class TreeEnsembleClassifier(TreeEnsembleClassifierCommon):

    atts = OrderedDict([
        ('base_values', numpy.empty(0, dtype=numpy.float32)),
        ('class_ids', numpy.empty(0, dtype=numpy.int64)),
        ('class_nodeids', numpy.empty(0, dtype=numpy.int64)),
        ('class_treeids', numpy.empty(0, dtype=numpy.int64)),
        ('class_weights', numpy.empty(0, dtype=numpy.float32)),
        ('classlabels_int64s', numpy.empty(0, dtype=numpy.int64)),
        ('classlabels_strings', []),
        ('nodes_falsenodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_featureids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_hitrates', numpy.empty(0, dtype=numpy.float32)),
        ('nodes_missing_value_tracks_true', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_modes', []),
        ('nodes_nodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_treeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_truenodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_values', numpy.empty(0, dtype=numpy.float32)),
        ('post_transform', b'NONE')
    ])

    def __init__(self, onnx_node, desc=None, **options):
        TreeEnsembleClassifierCommon.__init__(self, numpy.float32, onnx_node, desc=desc,
                                              expected_attributes=TreeEnsembleClassifier.atts,
                                              **options)


class TreeEnsembleClassifierDouble(TreeEnsembleClassifierCommon):

    atts = OrderedDict([
        ('base_values', numpy.empty(0, dtype=numpy.float64)),
        ('class_ids', numpy.empty(0, dtype=numpy.int64)),
        ('class_nodeids', numpy.empty(0, dtype=numpy.int64)),
        ('class_treeids', numpy.empty(0, dtype=numpy.int64)),
        ('class_weights', numpy.empty(0, dtype=numpy.float64)),
        ('classlabels_int64s', numpy.empty(0, dtype=numpy.int64)),
        ('classlabels_strings', []),
        ('nodes_falsenodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_featureids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_hitrates', numpy.empty(0, dtype=numpy.float64)),
        ('nodes_missing_value_tracks_true', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_modes', []),
        ('nodes_nodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_treeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_truenodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_values', numpy.empty(0, dtype=numpy.float64)),
        ('post_transform', b'NONE')
    ])

    def __init__(self, onnx_node, desc=None, **options):
        TreeEnsembleClassifierCommon.__init__(self, numpy.float64, onnx_node, desc=desc,
                                              expected_attributes=TreeEnsembleClassifier.atts,
                                              **options)


class TreeEnsembleClassifierDoubleSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl TreeEnsembleClassifierDouble.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'TreeEnsembleClassifierDouble')
        self.attributes = TreeEnsembleClassifierDouble.atts
