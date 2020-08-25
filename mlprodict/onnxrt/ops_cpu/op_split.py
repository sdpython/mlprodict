# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun
from ..shape_object import DimensionObject, ShapeObject


class Split(OpRun):
    """
    Runtime for operator *Split*.
    """

    atts = {'axis': 0, 'split': None}

    def __init__(self, onnx_node, desc=None, **options):
        if 'split' not in options:
            options['split'] = None
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Split.atts,
                       **options)
        self.nb_outputs = len(onnx_node.output)

    def _run(self, mat):  # pylint: disable=W0221
        if self.split is None:
            div = mat.shape[self.axis] // self.nb_outputs
            split = [div] * self.nb_outputs
            split[-1] += mat.shape[self.axis] - sum(split)
        else:
            split = self.split
        sli = [slice(0, s) for s in mat.shape]
        res = []
        pos = 0
        for spl in split:
            sli[self.axis] = slice(pos, pos + spl)
            pos += spl
            res.append(mat[tuple(sli)])
        return tuple(res)

    def _infer_shapes(self, data):  # pylint: disable=W0221
        if self.split is None:
            return tuple([ShapeObject(None, dtype=data.dtype)
                          for o in range(self.nb_outputs)])
        split = self.split

        res = []
        pos = 0
        for spl in split:
            shape = data.copy()
            shape[self.axis] = DimensionObject(spl)
            pos += spl
            res.append(shape)
        return tuple(res)
