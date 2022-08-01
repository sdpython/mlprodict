# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun


class SliceCommon(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, data, starts, ends, axes=None, steps=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(starts.shape) == 0:
            starts = numpy.array([starts])
        if len(ends.shape) == 0:
            ends = numpy.array([ends])
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
        try:
            return (data[tuple(slices)], )
        except TypeError as e:  # pragma: no cover
            raise TypeError(
                f"Unable to extract slice {slices!r} for shape {data.shape!r}.") from e


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

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return SliceCommon._run(
            self, data, self.starts, self.ends, self.axes)


if onnx_opset_version() >= 10:
    Slice = Slice_10
else:
    Slice = Slice_1  # pragma: no cover
