# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnary, RuntimeTypeError
from ._new_ops import OperatorSchema
from ..shape_object import ShapeObject


class Tokenizer(OpRunUnary):
    """
    See :epkg:`Tokenizer`.
    """

    atts = {'mark': 0,
            'mincharnum': 1,
            'pad_value': b'#',
            'separators': [' '],
            'tokenexp': b'[a-zA-Z0-9_]+',
            'stopwords': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=Tokenizer.atts,
                            **options)
        self.char_tokenization_ = (
            self.tokenexp == b'.' or self.separators == [b''])
        self.stops = set(_.decode() for _ in self.stopwords)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "Tokenizer":
            return TokenizerSchema()
        raise RuntimeError(
            "Unable to find a schema for operator '{}'.".format(op_name))

    def _run(self, text):  # pylint: disable=W0221
        if self.char_tokenization_:
            return self._run_char_tokenization(text, self.stops)
        raise NotImplementedError()

    def _run_char_tokenization(self, text, stops):
        max_len = max(map(len, text.flatten()))
        if self.mark:
            max_len += 2
            begin = 1
        else:
            begin = 0
        shape = text.shape + (max_len, )
        res = numpy.empty(shape, dtype=text.dtype)
        if len(text.shape) == 1:
            for i in range(text.shape[0]):
                pos = begin
                for j, c in enumerate(text[i]):
                    if c not in stops:
                        res[i, pos] = c
                        pos += 1
                if self.mark:
                    res[i, 0] = self.pad_value
                for j in range(pos, max_len):
                    res[i, j] = self.pad_value
        elif len(text.shape) == 2:
            for i in range(text.shape[0]):
                for ii in range(text.shape[1]):
                    pos = begin
                    for j, c in enumerate(text[i, ii]):
                        if c not in stops:
                            res[i, ii, pos] = c
                            pos += 1
                    if self.mark:
                        res[i, ii, 0] = self.pad_value
                    for j in range(pos, max_len):
                        res[i, ii, j] = self.pad_value
        else:
            raise RuntimeError(
                "Only vector or matrices are supported not shape {}.".format(text.shape))
        return (res, )

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


class TokenizerSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl TreeEnsembleClassifierDouble.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'Tokenizer')
        self.attributes = Tokenizer.atts
