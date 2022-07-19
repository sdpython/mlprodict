# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class DepthToSpace(OpRun):

    atts = {'blocksize': 0, 'mode': b'DCR'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=DepthToSpace.atts,
                       **options)

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(data.shape) != 4:
            raise RuntimeError(  # pragma: no cover
                f"Unexpected shape {data.shape!r}.")
        b, c, h, w = data.shape
        if self.mode == b'DCR':
            tmpshape = (b, self.blocksize, self.blocksize,
                        c // (self.blocksize * self.blocksize), h, w)
            reshaped = data.reshape(tmpshape)
            transposed = numpy.transpose(reshaped, [0, 3, 4, 1, 5, 2])
        else:
            # assert mode == "CRD"
            tmpshape = (b, c // (self.blocksize * self.blocksize),
                        self.blocksize, self.blocksize, h, w)
            reshaped = data.reshape(tmpshape)
            transposed = numpy.transpose(reshaped, [0, 1, 4, 2, 5, 3])
        finalshape = (b, c // (self.blocksize * self.blocksize),
                      h * self.blocksize, w * self.blocksize)
        y = numpy.reshape(transposed, finalshape)
        return (y, )


class SpaceToDepth(OpRun):

    atts = {'blocksize': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=SpaceToDepth.atts,
                       **options)

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(data.shape) != 4:
            raise RuntimeError(  # pragma: no cover
                f"Unexpected shape {data.shape!r}.")
        b, C, H, W = data.shape
        tmpshape = (b, C, H // self.blocksize, self.blocksize,
                    W // self.blocksize, self.blocksize)
        reshaped = numpy.reshape(data, tmpshape)
        transposed = numpy.transpose(reshaped, [0, 3, 5, 1, 2, 4])
        finalshape = (b, C * self.blocksize * self.blocksize,
                      H // self.blocksize, W // self.blocksize)
        y = numpy.reshape(transposed, finalshape)
        return (y, )
