# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from onnx.defs import onnx_opset_version
from ._op import OpRun
from ..shape_object import DimensionObject, ShapeObject


class CommonSplit(OpRun):
    """
    Runtime for operator *Split*.
    """

    def __init__(self, onnx_node, desc=None,
                 expected_attributes=None, **options):
        if 'split' not in options:
            options['split'] = None
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=expected_attributes,
                       **options)
        self.nb_outputs = len(onnx_node.output)

    def common_run(self, mat, split):  # pylint: disable=W0221
        if split is None:
            div = mat.shape[self.axis] // self.nb_outputs
            split = [div] * self.nb_outputs
            split[-1] += mat.shape[self.axis] - sum(split)
        sli = [slice(0, s) for s in mat.shape]
        res = []
        pos = 0
        for spl in split:
            sli[self.axis] = slice(pos, pos + spl)
            pos += spl
            res.append(mat[tuple(sli)])
        return tuple(res)

    def common_infer_shapes(self, data, split):  # pylint: disable=W0221
        if split is None:
            return tuple([ShapeObject(None, dtype=data.dtype)
                          for o in range(self.nb_outputs)])
        res = []
        pos = 0
        for spl in split:
            shape = data.copy()
            shape[self.axis] = DimensionObject(spl)
            pos += spl
            res.append(shape)
        return tuple(res)

    def _infer_types(self, data, split):  # pylint: disable=W0221
        if split is None:
            return tuple([data for o in range(self.nb_outputs)])
        return tuple(data for _ in split)

    def _infer_sizes(self, *args, **kwargs):  # pylint: disable=W0221
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res


class Split_2(CommonSplit):
    """
    Runtime for operator *Split*.
    """

    atts = {'axis': 0, 'split': None}

    def __init__(self, onnx_node, desc=None, **options):
        CommonSplit.__init__(self, onnx_node, desc=desc,
                             expected_attributes=Split_2.atts, **options)

    def _run(self, mat):  # pylint: disable=W0221
        return self.common_run(mat, self.split)

    def _infer_shapes(self, data):  # pylint: disable=W0221
        return self.common_infer_shapes(data, self.split)

    def _infer_types(self, data):  # pylint: disable=W0221
        if self.split is None:
            return tuple([data for o in range(self.nb_outputs)])
        return tuple(data for _ in self.split)


class Split_11(Split_2):
    """
    Runtime for operator *Split*.
    """
    pass


class Split_13(CommonSplit):
    """
    Runtime for operator *Split*.
    """

    atts = {'axis': 0}

    def __init__(self, onnx_node, desc=None, **options):
        CommonSplit.__init__(self, onnx_node, desc=desc,
                             expected_attributes=Split_13.atts, **options)

    def _run(self, mat, split=None):  # pylint: disable=W0221
        return self.common_run(mat, split)

    def _infer_shapes(self, data, split=None):  # pylint: disable=W0221
        return tuple([ShapeObject(None, dtype=data.dtype)
                      for o in range(self.nb_outputs)])

    def _infer_types(self, data, split=None):  # pylint: disable=W0221
        return tuple(data for o in range(self.nb_outputs))


if onnx_opset_version() >= 13:
    Split = Split_13
elif onnx_opset_version() >= 11:  # pragma: no cover
    Split = Split_11
else:  # pragma: no cover
    Split = Split_2
