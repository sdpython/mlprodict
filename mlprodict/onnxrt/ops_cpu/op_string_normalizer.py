# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import unicodedata
import locale
import warnings
import numpy
from ._op import OpRunUnary, RuntimeTypeError


class StringNormalizer(OpRunUnary):
    """
    The operator is not really threadsafe as python cannot
    play with two locales at the same time. stop words
    should not be implemented here as the tokenization
    usually happens after this steps.
    """

    atts = {'case_change_action': b'NONE',  # LOWER UPPER NONE
            'is_case_sensitive': 1,
            'locale': b'',
            'stopwords': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=StringNormalizer.atts,
                            **options)
        self.slocale = self.locale.decode('ascii')
        self.stops = set(self.stopwords)

    def _run(self, x):  # pylint: disable=W0221
        """
        Normalizes strings.
        """
        res = numpy.empty(x.shape, dtype=x.dtype)
        if len(x.shape) == 2:
            for i in range(0, x.shape[1]):
                self._run_column(x[:, i], res[:, i])
        elif len(x.shape) == 1:
            self._run_column(x, res)
        else:
            raise RuntimeTypeError(  # pragma: no cover
                "x must be a matrix or a vector.")
        return (res, )

    def _run_column(self, cin, cout):
        """
        Normalizes string in a columns.
        """
        if locale.getlocale() != self.slocale:
            try:
                locale.setlocale(locale.LC_ALL, self.slocale)
            except locale.Error as e:
                warnings.warn(
                    "Unknown local setting '{}' (current: '{}') - {}."
                    "".format(self.slocale, locale.getlocale(), e))
        stops = set(_.decode() for _ in self.stops)
        cout[:] = cin[:]

        for i in range(0, cin.shape[0]):
            cout[i] = self.strip_accents_unicode(cout[i])

        if self.is_case_sensitive and len(stops) > 0:
            for i in range(0, cin.shape[0]):
                cout[i] = self._remove_stopwords(cout[i], stops)

        if self.case_change_action == b'LOWER':
            for i in range(0, cin.shape[0]):
                cout[i] = cout[i].lower()
        elif self.case_change_action == b'UPPER':
            for i in range(0, cin.shape[0]):
                cout[i] = cout[i].upper()
        elif self.case_change_action != b'NONE':
            raise RuntimeError(
                "Unknown option for case_change_action: {}.".format(
                    self.case_change_action))

        if not self.is_case_sensitive and len(stops) > 0:
            for i in range(0, cin.shape[0]):
                cout[i] = self._remove_stopwords(cout[i], stops)

        return cout

    def _remove_stopwords(self, text, stops):
        spl = text.split(' ')
        return ' '.join(filter(lambda s: s not in stops, spl))

    def strip_accents_unicode(self, s):
        """
        Transforms accentuated unicode symbols into their simple counterpart.
        Source: `sklearn/feature_extraction/text.py
        <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/
        feature_extraction/text.py#L115>`_.

        :param s: string
            The string to strip
        :return: the cleaned string
        """
        try:
            # If `s` is ASCII-compatible, then it does not contain any accented
            # characters and we can avoid an expensive list comprehension
            s.encode("ASCII", errors="strict")
            return s
        except UnicodeEncodeError:
            normalized = unicodedata.normalize('NFKD', s)
            s = ''.join(
                [c for c in normalized if not unicodedata.combining(c)])
            return s

    def _infer_shapes(self, x):  # pylint: disable=E0202,W0221
        """
        Returns the same shape by default.
        """
        return (x, )
