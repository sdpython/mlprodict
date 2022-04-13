# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ..shape_object import ShapeObject
from ._op import OpRun
from .op_non_max_suppression_ import RuntimeNonMaxSuppression


class NonMaxSuppression(OpRun):

    atts = {'center_point_box': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=NonMaxSuppression.atts,
                       **options)
        self.inst = RuntimeNonMaxSuppression()
        self.inst.init(self.center_point_box)

    def _run(self, boxes, scores, max_output_boxes_per_class=None,  # pylint: disable=W0221
             iou_threshold=None, score_threshold=None):
        if max_output_boxes_per_class is None:
            max_output_boxes_per_class = numpy.array([], dtype=numpy.int64)
        if iou_threshold is None:
            iou_threshold = numpy.array([], dtype=numpy.float32)
        if score_threshold is None:
            score_threshold = numpy.array([], dtype=numpy.float32)
        return (self.inst.compute(
            boxes, scores, max_output_boxes_per_class,
            iou_threshold, score_threshold), )

    def _infer_shapes(self, boxes, scores, max_output_boxes_per_class=None,  # pylint: disable=W0221
                      iou_threshold=None, score_threshold=None):
        return (ShapeObject(None, dtype=numpy.int64), )
