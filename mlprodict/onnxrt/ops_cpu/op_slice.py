# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from onnx.defs import onnx_opset_version
from ..shape_object import ShapeObject
from ._op import OpRun


class SliceCommon(OpRun):

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
        if data.shape is None:
            return (ShapeObject(None, data.dtype), )
        shape = ["nslice%s_%d" % (pref, i) for i in range(len(data.shape))]
        return (ShapeObject(shape, data.dtype), )

    def _infer_types(self, data, starts, ends, axes=None, steps=None):  # pylint: disable=W0221
        return (data, )

    def _infer_sizes(self, *args, **kwargs):  # pylint: disable=W0221
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res


class Slice_10(SliceCommon):
    def __init__(self, onnx_node, desc=None, **options):
        SliceCommon.__init__(self, onnx_node, desc=desc,
                             **options)


class Slice_1(SliceCommon):

    atts = {'starts': [], 'ends': [], 'axes': []}

    def __init__(self, onnx_node, desc=None, **options):
        SliceCommon.__init__(self, onnx_node, desc=desc,
                             expected_attributes=Slice_1.atts,
                             **options)
        for f in ['starts', 'ends', 'steps', 'axes']:
            if not hasattr(self, f):
                continue
            if getattr(self, f) is not None and len(getattr(self, f)) == 0:
                setattr(self, f, None)

    def _run(self, data):  # pylint: disable=W0221
        return SliceCommon._run(
            self, data, self.starts, self.ends, self.axes)

    def _infer_shapes(self, data):  # pylint: disable=W0221
        return SliceCommon._infer_shapes(
            self, data, self.starts, self.ends, self.axes)

    def _infer_types(self, data):  # pylint: disable=W0221
        return (data, )


if onnx_opset_version() >= 10:
    Slice = Slice_10
else:
    Slice = Slice_1  # pragma: no cover
