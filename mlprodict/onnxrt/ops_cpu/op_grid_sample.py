# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from .op_grid_sample_ import GridSampleFloat, GridSampleDouble  # pylint: disable=E0611


class GridSample(OpRun):

    atts = {'align_corners': 0,
            'mode': b'bilinear',
            'padding_mode': b'zeros'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=GridSample.atts,
                       **options)
        self.rt32_ = None
        self.rt64_ = None
        self.rt32_ = GridSampleFloat()
        self.rt64_ = GridSampleDouble()
        self.rt32_.init(self.align_corners, self.mode, self.padding_mode)
        self.rt64_.init(self.align_corners, self.mode, self.padding_mode)

    def _run(self, X, grid, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if X.dtype == numpy.float32:
            if self.rt32_ is None:
                self.rt32_ = GridSampleFloat()
                self.rt32_.init(self.align_corners,
                                self.mode, self.padding_mode)
            rt = self.rt32_
        elif X.dtype == numpy.float32:
            if self.rt64_ is None:
                self.rt64_ = GridSampleDouble()
                self.rt64_.init(self.align_corners,
                                self.mode, self.padding_mode)
            rt = self.rt64_
        else:
            raise TypeError(  # pragma: no cover
                f"Unsupported type {X.dtype!r} for GridSample.")

        res = rt.compute(X, grid)
        return (res, )
