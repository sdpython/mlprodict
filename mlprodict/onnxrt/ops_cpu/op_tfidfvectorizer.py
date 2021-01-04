# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnary, RuntimeTypeError
from ..shape_object import ShapeObject
from .op_tfidfvectorizer_ import RuntimeTfIdfVectorizer  # pylint: disable=E0611,E0401


class TfIdfVectorizer(OpRunUnary):

    atts = {'max_gram_length': 1,
            'max_skip_count': 1,
            'min_gram_length': 1,
            'mode': b'TF',
            'ngram_counts': [],
            'ngram_indexes': [],
            'pool_int64s': [],
            'pool_strings': [],
            'weights': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=TfIdfVectorizer.atts,
                            **options)
        self.rt_ = RuntimeTfIdfVectorizer()
        if len(self.pool_strings) != 0:
            pool_int64s = list(range(len(self.pool_strings)))
            pool_strings_ = numpy.array(
                [_.decode('utf-8') for _ in self.pool_strings])
            mapping = {}
            for i, w in enumerate(pool_strings_):
                mapping[w] = i
        else:
            mapping = None
            pool_int64s = self.pool_int64s
            pool_strings_ = None

        self.mapping_ = mapping
        self.pool_strings_ = pool_strings_
        self.rt_.init(
            self.max_gram_length, self.max_skip_count, self.min_gram_length,
            self.mode, self.ngram_counts, self.ngram_indexes, pool_int64s,
            self.weights)

    def _run(self, x):  # pylint: disable=W0221
        if self.mapping_ is None:
            res = self.rt_.compute(x)
            return (res.reshape((x.shape[0], -1)), )
        else:
            xi = numpy.empty(x.shape, dtype=numpy.int64)
            for i in range(0, x.shape[0]):
                for j in range(0, x.shape[1]):
                    try:
                        xi[i, j] = self.mapping_[x[i, j]]
                    except KeyError:
                        xi[i, j] = -1
            res = self.rt_.compute(xi)
            return (res.reshape((x.shape[0], -1)), )

    def _infer_shapes(self, x):  # pylint: disable=E0202,W0221
        """
        Returns the same shape by default.
        """
        if x.shape is None:
            return (x, )
        if len(x) == 1:
            return (ShapeObject((x[0], None), dtype=x.dtype,
                                name=self.__class__.__name__), )
        if len(x) == 2:
            return (ShapeObject((x[0], x[1], None), dtype=x.dtype,
                                name=self.__class__.__name__), )
        raise RuntimeTypeError(
            "Only two dimension are allowed, got {}.".format(x))
