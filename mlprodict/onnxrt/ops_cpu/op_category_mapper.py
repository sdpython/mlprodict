# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class CategoryMapper(OpRun):

    atts = {'cats_int64s': numpy.empty(0, dtype=numpy.int64),
            'cats_strings': numpy.empty(0, dtype=numpy.str_),
            'default_int64': -1,
            'default_string': b'',
            }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=CategoryMapper.atts,
                       **options)
        if len(self.cats_int64s) != len(self.cats_strings):
            raise RuntimeError(  # pragma: no cover
                "Lengths mismatch between cats_int64s (%d) and "
                "cats_strings (%d)." % (
                    len(self.cats_int64s), len(self.cats_strings)))
        self.int2str_ = {}
        self.str2int_ = {}
        for a, b in zip(self.cats_int64s, self.cats_strings):
            be = b.decode('utf-8')
            self.int2str_[a] = be
            self.str2int_[be] = a

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if x.dtype == numpy.int64:
            xf = x.ravel()
            res = [self.int2str_.get(xf[i], self.default_string)
                   for i in range(0, xf.shape[0])]
            return (numpy.array(res).reshape(x.shape), )

        xf = x.ravel()
        res = numpy.empty((xf.shape[0], ), dtype=numpy.int64)
        for i in range(0, res.shape[0]):
            res[i] = self.str2int_.get(xf[i], self.default_int64)
        return (res.reshape(x.shape), )
