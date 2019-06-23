# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from collections import OrderedDict
import numpy
from ._op import OpRun
from .op_tree_ensemble_classifier_ import RuntimeTreeEnsembleClassifier  # pylint: disable=E0611


class TreeEnsembleClassifier(OpRun):

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
        ('post_transform', 'NONE')
    ])

    def __init__(self, onnx_node, desc=None, **options):
        if desc is None:
            raise ValueError("desc should not be None.")
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=TreeEnsembleClassifier.atts,
                       **options)
        self._init()

    def _get_typed_attributes(self, k):
        ty = TreeEnsembleClassifier.atts[k]
        if isinstance(ty, numpy.ndarray):
            return getattr(self, k).astype(ty.dtype)
        elif isinstance(ty, str):
            return getattr(self, k).decode()
        elif isinstance(ty, list):
            return [_.decode() for _ in getattr(self, k)]
        else:
            raise NotImplementedError("Unable to convert '{}' ({}).".format(
                k, getattr(self, k)))

    def _init(self):
        self.rt_ = RuntimeTreeEnsembleClassifier()
        atts = [self._get_typed_attributes(k)
                for k in TreeEnsembleClassifier.atts]
        self.rt_.init(*atts)

    def _run(self, x):  # pylint: disable=W0221
        label, scores = self.rt_.compute(x)
        scores = scores.reshape(label.shape[0],
                                scores.shape[0] // label.shape[0])
        return (label, scores)
