# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from scipy.sparse import coo_matrix
from ._op import OpRun, RuntimeTypeError


class DictVectorizer(OpRun):

    atts = {'int64_vocabulary': numpy.empty(0, dtype=numpy.int64),
            'string_vocabulary': numpy.empty(0, dtype=numpy.str_)}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=DictVectorizer.atts,
                       **options)
        self.dict_labels = {}
        if len(self.int64_vocabulary) > 0:
            for i, v in enumerate(self.int64_vocabulary):
                self.dict_labels[v] = i
            self.is_int = True
        else:
            for i, v in enumerate(self.string_vocabulary):
                self.dict_labels[v.decode('utf-8')] = i
            self.is_int = False
        if len(self.dict_labels) == 0:
            raise RuntimeError(  # pragma: no cover
                "int64_vocabulary and string_vocabulary cannot be both empty.")

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if not isinstance(x, (numpy.ndarray, list)):
            raise RuntimeTypeError(  # pragma: no cover
                f"x must be iterable not {type(x)}.")
        values = []
        rows = []
        cols = []
        for i, row in enumerate(x):
            for k, v in row.items():
                values.append(v)
                rows.append(i)
                cols.append(self.dict_labels[k])
        values = numpy.array(values)
        rows = numpy.array(rows)
        cols = numpy.array(cols)
        return (coo_matrix((values, (rows, cols)), shape=(len(x), len(self.dict_labels))), )
