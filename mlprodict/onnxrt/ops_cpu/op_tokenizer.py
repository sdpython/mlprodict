# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import re
import numpy
from ._op import OpRunUnary, RuntimeTypeError
from ._new_ops import OperatorSchema


class Tokenizer(OpRunUnary):
    """
    See :epkg:`Tokenizer`.
    """

    atts = {'mark': 0,
            'mincharnum': 1,
            'pad_value': b'#',
            'separators': [],
            'tokenexp': b'[a-zA-Z0-9_]+',
            'tokenexpsplit': 0,
            'stopwords': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=Tokenizer.atts,
                            **options)
        self.char_tokenization_ = (
            self.tokenexp == b'.' or list(self.separators) == [b''])
        self.stops_ = set(_.decode() for _ in self.stopwords)
        try:
            self.str_separators_ = set(_.decode('utf-8')
                                       for _ in self.separators)
        except AttributeError as e:  # pragma: no cover
            raise RuntimeTypeError(
                f"Unable to interpret separators {self.separators}.") from e
        if self.tokenexp not in (None, b''):
            self.tokenexp_ = re.compile(self.tokenexp.decode('utf-8'))

    def _find_custom_operator_schema(self, op_name):
        if op_name == "Tokenizer":
            return TokenizerSchema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")

    def _run(self, text, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.char_tokenization_:
            return self._run_char_tokenization(text, self.stops_)
        if self.str_separators_ is not None and len(self.str_separators_) > 0:
            return self._run_sep_tokenization(
                text, self.stops_, self.str_separators_)
        if self.tokenexp not in (None, ''):
            return self._run_regex_tokenization(
                text, self.stops_, self.tokenexp_)
        raise RuntimeError(  # pragma: no cover
            "Unable to guess which tokenization to use, sep={}, "
            "tokenexp='{}'.".format(self.separators, self.tokenexp))

    def _run_tokenization(self, text, stops, split):
        """
        Tokenizes a char level.
        """
        max_len = max(map(len, text.flatten()))
        if self.mark:
            max_len += 2
            begin = 1
        else:
            begin = 0
        shape = text.shape + (max_len, )
        max_pos = 0
        res = numpy.empty(shape, dtype=text.dtype)
        if len(text.shape) == 1:
            res[:] = self.pad_value
            for i in range(text.shape[0]):
                pos = begin
                for c in split(text[i]):
                    if c not in stops:
                        res[i, pos] = c
                        pos += 1
                if self.mark:
                    res[i, 0] = self.pad_value
                    max_pos = max(pos + 1, max_pos)
                else:
                    max_pos = max(pos, max_pos)
            res = res[:, :max_pos]
        elif len(text.shape) == 2:
            res[:, :] = self.pad_value
            for i in range(text.shape[0]):
                for ii in range(text.shape[1]):
                    pos = begin
                    for c in split(text[i, ii]):
                        if c not in stops:
                            res[i, ii, pos] = c
                            pos += 1
                    if self.mark:
                        res[i, ii, 0] = self.pad_value
                        max_pos = max(pos + 1, max_pos)
                    else:
                        max_pos = max(pos, max_pos)
            res = res[:, :, :max_pos]
        else:
            raise RuntimeError(  # pragma: no cover
                f"Only vector or matrices are supported not shape {text.shape}.")
        return (res, )

    def _run_char_tokenization(self, text, stops):
        """
        Tokenizes y charaters.
        """
        def split(t):
            for c in t:
                yield c
        return self._run_tokenization(text, stops, split)

    def _run_sep_tokenization(self, text, stops, separators):
        """
        Tokenizes using separators.
        The function should use a trie to find text.
        """
        def split(t):
            begin = 0
            pos = 0
            while pos < len(t):
                for sep in separators:
                    if (pos + len(sep) <= len(t) and
                            sep == t[pos: pos + len(sep)]):
                        word = t[begin: pos]
                        yield word
                        begin = pos + len(sep)
                        break
                pos += 1
            if begin < pos:
                word = t[begin: pos]
                yield word

        return self._run_tokenization(text, stops, split)

    def _run_regex_tokenization(self, text, stops, exp):
        """
        Tokenizes using separators.
        The function should use a trie to find text.
        """
        if self.tokenexpsplit:
            def split(t):
                return filter(lambda x: x, exp.split(t))
        else:
            def split(t):
                return filter(lambda x: x, exp.findall(t))
        return self._run_tokenization(text, stops, split)


class TokenizerSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl TreeEnsembleClassifierDouble.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'Tokenizer')
        self.attributes = Tokenizer.atts
