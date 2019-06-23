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
from .op_tree_ensemble_regressor_ import RuntimeTreeEnsembleRegressor  # pylint: disable=E0611


class TreeEnsembleRegressor(OpRun):

    atts = OrderedDict([
        ('aggregate_function', b'SUM'),
        ('base_values', numpy.empty(0, dtype=numpy.float32)),
        ('n_targets', 1),
        ('nodes_falsenodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_featureids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_hitrates', numpy.empty(0, dtype=numpy.float32)),
        ('nodes_missing_value_tracks_true', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_modes', []),
        ('nodes_nodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_treeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_truenodeids', numpy.empty(0, dtype=numpy.int64)),
        ('nodes_values', numpy.empty(0, dtype=numpy.float32)),
        ('post_transform', b'NONE'),
        ('target_ids', numpy.empty(0, dtype=numpy.int64)),
        ('target_nodeids', numpy.empty(0, dtype=numpy.int64)),
        ('target_treeids', numpy.empty(0, dtype=numpy.int64)),
        ('target_weights', numpy.empty(0, dtype=numpy.float32)),
    ])

    def __init__(self, onnx_node, desc=None, **options):
        if desc is None:
            raise ValueError("desc should not be None.")
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=TreeEnsembleRegressor.atts,
                       **options)
        self._init()

    def _get_typed_attributes(self, k):
        return _get_typed_class_attribute(self, k, TreeEnsembleRegressor.atts)

    def _init(self):
        self.rt_ = RuntimeTreeEnsembleRegressor()
        atts = [self._get_typed_attributes(k)
                for k in TreeEnsembleRegressor.atts]
        self.rt_.init(*atts)

    def _run(self, x):  # pylint: disable=W0221
        pred = self.rt_.compute(x)
        if pred.shape[0] != x.shape[0]:
            pred = pred.reshape(x.shape[0], pred.shape[0] // x.shape[0])
        return (pred, )
