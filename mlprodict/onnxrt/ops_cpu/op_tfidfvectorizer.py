# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnary
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
            pool_strings_ = numpy.array(
                [_.decode('utf-8') for _ in self.pool_strings])
            mapping = {}
            pool_int64s = []
            for i, w in enumerate(pool_strings_):
                if w not in mapping:
                    # 1-gram are processed first.
                    mapping[w] = i
                pool_int64s.append(mapping[w])
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

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.mapping_ is None:
            res = self.rt_.compute(x)
            if len(x.shape) > 1:
                return (res.reshape((x.shape[0], -1)), )
            return (res, )

        xi = numpy.empty(x.shape, dtype=numpy.int64)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                try:
                    xi[i, j] = self.mapping_[x[i, j]]
                except KeyError:
                    xi[i, j] = -1
        res = self.rt_.compute(xi)
        return (res.reshape((x.shape[0], -1)), )
