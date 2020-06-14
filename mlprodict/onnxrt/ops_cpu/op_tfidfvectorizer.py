# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRunUnary, RuntimeTypeError
from ..shape_object import ShapeObject


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

    def _run(self, x):  # pylint: disable=W0221
        raise NotImplementedError()

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
