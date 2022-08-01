# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from collections import OrderedDict
import numpy
from onnx.defs import onnx_opset_version
from ._op_helper import _get_typed_class_attribute
from ._op import OpRunClassifierProb, RuntimeTypeError
from ._op_classifier_string import _ClassifierCommon
from ._new_ops import OperatorSchema
from .op_tree_ensemble_classifier_ import (  # pylint: disable=E0611,E0401
    RuntimeTreeEnsembleClassifierDouble,
    RuntimeTreeEnsembleClassifierFloat)
from .op_tree_ensemble_classifier_p_ import (  # pylint: disable=E0611,E0401
    RuntimeTreeEnsembleClassifierPFloat,
    RuntimeTreeEnsembleClassifierPDouble)


class TreeEnsembleClassifierCommon(OpRunClassifierProb, _ClassifierCommon):

    def __init__(self, dtype, onnx_node, desc=None,
                 expected_attributes=None,
                 runtime_version=3, **options):
        OpRunClassifierProb.__init__(
            self, onnx_node, desc=desc,
            expected_attributes=expected_attributes, **options)
        self._init(dtype=dtype, version=runtime_version)

    def _get_typed_attributes(self, k):
        return _get_typed_class_attribute(self, k, self.__class__.atts)

    def _find_custom_operator_schema(self, op_name):
        """
        Finds a custom operator defined by this runtime.
        """
        if op_name == "TreeEnsembleClassifierDouble":
            return TreeEnsembleClassifierDoubleSchema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")

    def _init(self, dtype, version):
        self._post_process_label_attributes()

        atts = []
        for k in self.__class__.atts:
            v = self._get_typed_attributes(k)
            if k.endswith('_as_tensor'):
                if (v is not None and isinstance(v, numpy.ndarray) and
                        v.size > 0):
                    # replacements
                    atts[-1] = v
                    if dtype is None:
                        dtype = v.dtype
                continue
            atts.append(v)

        if dtype is None:
            dtype = numpy.float32

        if dtype == numpy.float32:
            if version == 0:
                self.rt_ = RuntimeTreeEnsembleClassifierFloat()
            elif version == 1:
                self.rt_ = RuntimeTreeEnsembleClassifierPFloat(
                    60, 20, False, False)
            elif version == 2:
                self.rt_ = RuntimeTreeEnsembleClassifierPFloat(
                    60, 20, True, False)
            elif version == 3:
                self.rt_ = RuntimeTreeEnsembleClassifierPFloat(
                    60, 20, True, True)
            else:
                raise ValueError(f"Unknown version '{version}'.")
        elif dtype == numpy.float64:
            if version == 0:
                self.rt_ = RuntimeTreeEnsembleClassifierDouble()
            elif version == 1:
                self.rt_ = RuntimeTreeEnsembleClassifierPDouble(
                    60, 20, False, False)
            elif version == 2:
                self.rt_ = RuntimeTreeEnsembleClassifierPDouble(
                    60, 20, True, False)
            elif version == 3:
                self.rt_ = RuntimeTreeEnsembleClassifierPDouble(
                    60, 20, True, True)
            else:
                raise ValueError(  # pragma: no cover
                    f"Unknown version '{version}'.")
        else:
            raise RuntimeTypeError(  # pragma: no cover
                f"Unsupported dtype={dtype}.")
        self.rt_.init(*atts)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        """
        This is a C++ implementation coming from
        :epkg:`onnxruntime`.
        `tree_ensemble_classifier.cc
        <https://github.com/microsoft/onnxruntime/blob/master/
        onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc>`_.
        See class :class:`RuntimeTreeEnsembleClassifier
        <mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_.RuntimeTreeEnsembleClassifier>`.
        """
        label, scores = self.rt_.compute(x)
        if scores.shape[0] != label.shape[0]:
            scores = scores.reshape(label.shape[0],
                                    scores.shape[0] // label.shape[0])
        return self._post_process_predicted_label(label, scores)


class TreeEnsembleClassifier_1(TreeEnsembleClassifierCommon):

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
        TreeEnsembleClassifierCommon.__init__(
            self, numpy.float32, onnx_node, desc=desc,
            expected_attributes=TreeEnsembleClassifier_1.atts, **options)


class TreeEnsembleClassifier_3(TreeEnsembleClassifierCommon):

    atts = OrderedDict([
        ('base_values', numpy.empty(0, dtype=numpy.float32)),
        ('base_values_as_tensor', []),
        ('class_ids', numpy.empty(0, dtype=numpy.int64)),
        ('class_nodeids', numpy.empty(0, dtype=numpy.int64)),
        ('class_treeids', numpy.empty(0, dtype=numpy.int64)),
        ('class_weights', numpy.empty(0, dtype=numpy.float32)),
        ('class_weights_as_tensor', []),
        ('classlabels_int64s', numpy.empty(0, dtype=numpy.int64)),
        ('classlabels_strings', []),
        ('nodes_falsenodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_featureids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_hitrates', numpy.empty(0, dtype=numpy.float32)),
        ('nodes_hitrates_as_tensor', []),
        ('nodes_missing_value_tracks_true', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_modes', []),
        ('nodes_nodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_treeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_truenodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_values', numpy.empty(0, dtype=numpy.float32)),
        ('nodes_values_as_tensor', []),
        ('post_transform', b'NONE')
    ])

    def __init__(self, onnx_node, desc=None, **options):
        TreeEnsembleClassifierCommon.__init__(
            self, None, onnx_node, desc=desc,
            expected_attributes=TreeEnsembleClassifier_3.atts, **options)


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
        TreeEnsembleClassifierCommon.__init__(
            self, numpy.float64, onnx_node, desc=desc,
            expected_attributes=TreeEnsembleClassifierDouble.atts, **options)


class TreeEnsembleClassifierDoubleSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl TreeEnsembleClassifierDouble.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'TreeEnsembleClassifierDouble')
        self.attributes = TreeEnsembleClassifierDouble.atts


if onnx_opset_version() >= 16:
    TreeEnsembleClassifier = TreeEnsembleClassifier_3
else:
    TreeEnsembleClassifier = TreeEnsembleClassifier_1
