# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from .op_roi_align_ import RoiAlignFloat, RoiAlignDouble  # pylint: disable=E0611


class RoiAlign(OpRun):

    atts = {'coordinate_transformation_mode': b'half_pixel',
            'mode': b'avg',
            'output_height': 1,
            'output_width': 1,
            'sampling_ratio': 0,
            'spatial_scale': 1.}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=RoiAlign.atts,
                       **options)
        self.rt32_ = None
        self.rt64_ = None

    def _run(self, X, rois, batch_indices, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if X.dtype == numpy.float32:
            if self.rt32_ is None:
                self.rt32_ = RoiAlignFloat()
                self.rt32_.init(
                    self.coordinate_transformation_mode.decode('ascii'),
                    self.mode.decode('ascii'), self.output_height,
                    self.output_width, self.sampling_ratio, self.spatial_scale)
            rt = self.rt32_
        elif X.dtype == numpy.float64:
            if self.rt64_ is None:
                self.rt64_ = RoiAlignDouble()
                self.rt64_.init(
                    self.coordinate_transformation_mode.decode('ascii'),
                    self.mode.decode('ascii'), self.output_height,
                    self.output_width, self.sampling_ratio, self.spatial_scale)
            rt = self.rt64_
        else:
            raise TypeError(
                f"Unexpected type {X.dtype!r} for X.")

        res = rt.compute(X, rois, batch_indices)
        return (res, )
