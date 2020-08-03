# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import DimensionObject


class OneHotEncoder(OpRun):
    """
    :epkg:`ONNX` specifications does not mention
    the possibility to change the output type,
    sparse, dense, float, double.
    """

    atts = {'cats_int64s': numpy.empty(0, dtype=numpy.int64),
            'cats_strings': numpy.empty(0, dtype=numpy.str),
            'zeros': 1,
            }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=OneHotEncoder.atts,
                       **options)
        if len(self.cats_int64s) > 0:
            self.classes_ = {v: i for i, v in enumerate(self.cats_int64s)}
        elif len(self.cats_strings) > 0:
            self.classes_ = {v.decode('utf-8'): i for i,
                             v in enumerate(self.cats_strings)}
        else:
            raise RuntimeError("No encoding was defined.")  # pragma: no cover

    def _run(self, x):  # pylint: disable=W0221
        shape = x.shape
        new_shape = shape + (len(self.classes_), )
        res = numpy.zeros(new_shape, dtype=numpy.float32)
        if len(x.shape) == 1:
            for i, v in enumerate(x):
                j = self.classes_.get(v, -1)
                if j >= 0:
                    res[i, j] = 1.
        elif len(x.shape) == 2:
            for a, row in enumerate(x):
                for i, v in enumerate(row):
                    j = self.classes_.get(v, -1)
                    if j >= 0:
                        res[a, i, j] = 1.
        else:
            raise RuntimeError(  # pragma: no cover
                "This operator is not implemented for shape {}.".format(x.shape))

        if not self.zeros:
            red = res.sum(axis=len(res.shape) - 1)
            if numpy.min(red) == 0:
                rows = []
                for i, val in enumerate(red):
                    if val == 0:
                        rows.append(dict(row=i, value=x[i]))
                        if len(rows) > 5:
                            break
                raise RuntimeError(  # pragma no cover
                    "One observation did not have any defined category.\n"
                    "classes: {}\nfirst rows:\n{}\nres:\n{}\nx:\n{}".format(
                        self.classes_, "\n".join(str(_) for _ in rows),
                        res[:5], x[:5]))

        return (res, )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        new_shape = x.copy()
        dim = DimensionObject(len(self.classes_))
        new_shape.append(dim)
        new_shape._dtype = numpy.float32
        new_shape.name = self.onnx_node.name
        return (new_shape, )
