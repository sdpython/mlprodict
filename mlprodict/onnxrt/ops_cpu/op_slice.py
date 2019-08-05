# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun
from ..shape_object import ShapeObject


class Slice(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, data, starts, ends, axes=None, steps=None):  # pylint: disable=W0221
        if axes is None:
            if steps is None:
                slices = [slice(s, e) for s, e in zip(starts, ends)]
            else:
                slices = [slice(s, e, d)
                          for s, e, d in zip(starts, ends, steps)]
        else:
            if steps is None:
                slices = [slice(0, a) for a in data.shape]
                for s, e, a in zip(starts, ends, axes):
                    slices[a] = slice(s, e)
            else:
                slices = [slice(0, a) for a in data.shape]
                for s, e, a, d in zip(starts, ends, axes, steps):
                    slices[a] = slice(s, e, d)
        return (data[tuple(slices)], )

    def _infer_shapes(self, data, starts, ends, axes=None, steps=None):  # pylint: disable=W0221
        pref = str(hex(id(self))[2:])
        shape = ["nslice%s_%d" % (pref, i) for i in range(len(data))]
        return (ShapeObject(shape, data.dtype), )
