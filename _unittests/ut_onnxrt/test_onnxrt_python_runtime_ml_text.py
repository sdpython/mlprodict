# coding: utf-8
"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx.common.data_types import (
    StringTensorType, FloatTensorType, Int64TensorType)
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxStringNormalizer, OnnxTfIdfVectorizer)
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnx_conv.onnx_ops import OnnxTokenizer
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools import get_opset_number_from_onnx


class TestOnnxrtPythonRuntimeMlText(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxrt_string_normalizer(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?'])

        op = OnnxStringNormalizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'])
        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(list(res['out']), list(corpus))

        res = oinf.run({'text': corpus.reshape((2, 2))})
        self.assertEqual(res['out'].tolist(), corpus.reshape((2, 2)).tolist())

        op = OnnxStringNormalizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'],
            case_change_action='LOWER')
        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(list(res['out']), list(_.lower() for _ in corpus))

        op = OnnxStringNormalizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'],
            case_change_action='UPPER')
        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(list(res['out']), list(_.upper() for _ in corpus))

        op = OnnxStringNormalizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'],
            case_change_action='UPPER2')
        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        oinf = OnnxInference(onx)
        self.assertRaise(lambda: oinf.run({'text': corpus}), RuntimeError)

    def test_onnxrt_string_normalizer_stopwords(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?'])

        op = OnnxStringNormalizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'], stopwords=['this'])
        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(
            list(res['out']), list(_.replace("this ", "") for _ in corpus))

        op = OnnxStringNormalizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'], stopwords=['this'],
            case_change_action='LOWER', is_case_sensitive=0)
        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(
            list(res['out']), list(_.lower().replace("this ", "") for _ in corpus))

    def test_onnxrt_string_normalizer_stopwords_french(self):
        corpus = numpy.array([
            'A is the first document.',
            'This document is the second document.',
            'And a is the third one.',
            'Is à the first document?'])
        exp = numpy.array([
            'a is the first document.',
            'this document is the second document.',
            'and a is the third one.',
            'is a the first document?'])

        op = OnnxStringNormalizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'], case_change_action='LOWER',
            locale='fr_FR')
        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(list(res['out']), list(exp))

    def test_onnxrt_string_normalizer_empty(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?'])

        op = OnnxStringNormalizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'])
        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        oinf = OnnxInference(onx)
        corpus[-1] = ""
        res = oinf.run({'text': corpus})
        self.assertEqual(list(res['out']), list(corpus))

    def test_onnxrt_tokenizer_char(self):
        corpus = numpy.array(['abc', 'abc d', 'abc  e'])
        exp = numpy.array(
            [['a', 'b', 'c', '#', '#', '#'],
             ['a', 'b', 'c', ' ', 'd', '#'],
             ['a', 'b', 'c', ' ', ' ', 'e']])

        op = OnnxTokenizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'], tokenexp='.')
        onx = op.to_onnx(inputs=[('text', StringTensorType())],
                         outputs=[('out', StringTensorType())])
        self.assertIn('domain: "mlprodict"', str(onx))
        self.assertIn('version: 1', str(onx))
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(res['out'].tolist(), exp.tolist())
        res = oinf.run({'text': corpus.reshape((-1, 1))})
        self.assertEqual(res['out'].tolist(), exp.reshape((3, 1, -1)).tolist())

    def test_onnxrt_tokenizer_char_mark(self):
        corpus = numpy.array(['abc', 'abc d', 'abc  e'])
        exp = numpy.array(
            [['#', 'a', 'b', 'c', '#', '#', '#', '#'],
             ['#', 'a', 'b', 'c', ' ', 'd', '#', '#'],
             ['#', 'a', 'b', 'c', ' ', ' ', 'e', '#']])

        op = OnnxTokenizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'], tokenexp='.', mark=1)
        onx = op.to_onnx(inputs=[('text', StringTensorType())],
                         outputs=[('out', StringTensorType())])
        self.assertIn('domain: "mlprodict"', str(onx))
        self.assertIn('version: 1', str(onx))
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(res['out'].tolist(), exp.tolist())

    def test_onnxrt_tokenizer_word_mark(self):
        corpus = numpy.array(['abc ef zoo', 'abc,d', 'ab/e'])
        exp = numpy.array(
            [['#', 'abc', 'ef', 'zoo', '#'],
             ['#', 'abc', 'd', '#', '#'],
             ['#', 'ab', 'e', '#', '#']])

        op = OnnxTokenizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'], separators=[' ', ',', '/'], mark=1)
        onx = op.to_onnx(inputs=[('text', StringTensorType())],
                         outputs=[('out', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(res['out'].tolist(), exp.tolist())

    def test_onnxrt_tokenizer_word_stop(self):
        corpus = numpy.array(['abc ef zoo', 'abc,d', 'ab/e'])
        exp = numpy.array(
            [['abc', 'ef', 'zoo'],
             ['abc', '#', '#'],
             ['ab', 'e', '#']])

        op = OnnxTokenizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'], separators=[' ', ',', '/'], mark=0,
            stopwords=['d'])
        onx = op.to_onnx(inputs=[('text', StringTensorType())],
                         outputs=[('out', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(res['out'].tolist(), exp.tolist())

    def test_onnxrt_tokenizer_word_regex_mark_split(self):
        corpus = numpy.array(['abc ef zoo', 'abc,d', 'ab/e'])
        exp = numpy.array(
            [['#', ' ef zoo', '#'],
             ['#', ',d', '#'],
             ['#', '/e', '#']])

        op = OnnxTokenizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'], mark=1, tokenexp='[a-c]+',
            tokenexpsplit=1)
        onx = op.to_onnx(inputs=[('text', StringTensorType())],
                         outputs=[('out', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(res['out'].tolist(), exp.tolist())

    def test_onnxrt_tokenizer_word_regex_mark_findall(self):
        corpus = numpy.array(['abc ef zoo', 'abc,d', 'ab/e'])
        exp = numpy.array(
            [['#', 'abc', '#'],
             ['#', 'abc', '#'],
             ['#', 'ab', '#']])

        op = OnnxTokenizer(
            'text', op_version=get_opset_number_from_onnx(),
            output_names=['out'], mark=1, tokenexp='[a-c]+',
            tokenexpsplit=0)
        onx = op.to_onnx(inputs=[('text', StringTensorType())],
                         outputs=[('out', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(res['out'].tolist(), exp.tolist())

    def test_onnxrt_tfidf_vectorizer(self):
        inputi = numpy.array([[1, 1, 3, 3, 3, 7],
                              [8, 6, 7, 5, 6, 8]]).astype(numpy.int64)
        output = numpy.array([[0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 1.]]).astype(numpy.float32)

        ngram_counts = numpy.array([0, 4]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2, 3, 4, 5, 6]).astype(numpy.int64)
        pool_int64s = numpy.array([2, 3, 5, 4,    # unigrams
                                   5, 6, 7, 8, 6, 7]).astype(numpy.int64)   # bigrams

        op = OnnxTfIdfVectorizer(
            'tokens', op_version=get_opset_number_from_onnx(),
            mode='TF', min_gram_length=2, max_gram_length=2,
            max_skip_count=0, ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes, pool_int64s=pool_int64s,
            output_names=['out'])
        onx = op.to_onnx(inputs=[('tokens', Int64TensorType())],
                         outputs=[('out', FloatTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'tokens': inputi})
        self.assertEqual(output.tolist(), res['out'].tolist())

    def test_onnxrt_tfidf_vectorizer_skip5(self):
        inputi = numpy.array([[1, 1, 3, 3, 3, 7],
                              [8, 6, 7, 5, 6, 8]]).astype(numpy.int64)
        output = numpy.array([[0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 1., 1.]]).astype(numpy.float32)

        ngram_counts = numpy.array([0, 4]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2, 3, 4, 5, 6]).astype(numpy.int64)
        pool_int64s = numpy.array([2, 3, 5, 4,    # unigrams
                                   5, 6, 7, 8, 6, 7]).astype(numpy.int64)   # bigrams

        op = OnnxTfIdfVectorizer(
            'tokens', op_version=get_opset_number_from_onnx(),
            mode='TF', min_gram_length=2, max_gram_length=2,
            max_skip_count=5, ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes, pool_int64s=pool_int64s,
            output_names=['out'])
        onx = op.to_onnx(inputs=[('tokens', Int64TensorType())],
                         outputs=[('out', FloatTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'tokens': inputi})
        self.assertEqual(output.tolist(), res['out'].tolist())

    def test_onnxrt_tfidf_vectorizer_unibi_skip5(self):
        inputi = numpy.array([[1, 1, 3, 3, 3, 7],
                              [8, 6, 7, 5, 6, 8]]).astype(numpy.int64)
        output = numpy.array([[0., 3., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 1., 1., 1.]]).astype(numpy.float32)

        ngram_counts = numpy.array([0, 4]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2, 3, 4, 5, 6]).astype(numpy.int64)
        pool_int64s = numpy.array([2, 3, 5, 4,    # unigrams
                                   5, 6, 7, 8, 6, 7]).astype(numpy.int64)   # bigrams

        op = OnnxTfIdfVectorizer(
            'tokens', op_version=get_opset_number_from_onnx(),
            mode='TF', min_gram_length=1, max_gram_length=2,
            max_skip_count=5, ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes, pool_int64s=pool_int64s,
            output_names=['out'])
        onx = op.to_onnx(inputs=[('tokens', Int64TensorType())],
                         outputs=[('out', FloatTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'tokens': inputi})
        self.assertEqual(output.tolist(), res['out'].tolist())

    def test_onnxrt_tfidf_vectorizer_bi_skip0(self):
        inputi = numpy.array(
            [[1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]]).astype(numpy.int64)
        output = numpy.array([[0., 0., 0., 0., 1., 1., 1.]]
                             ).astype(numpy.float32)

        ngram_counts = numpy.array([0, 4]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2, 3, 4, 5, 6]).astype(numpy.int64)
        pool_int64s = numpy.array([2, 3, 5, 4,    # unigrams
                                   5, 6, 7, 8, 6, 7]).astype(numpy.int64)   # bigrams

        op = OnnxTfIdfVectorizer(
            'tokens', op_version=get_opset_number_from_onnx(),
            mode='TF', min_gram_length=2, max_gram_length=2,
            max_skip_count=0, ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes, pool_int64s=pool_int64s,
            output_names=['out'])
        onx = op.to_onnx(inputs=[('tokens', Int64TensorType())],
                         outputs=[('out', FloatTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'tokens': inputi})
        self.assertEqual(output.tolist(), res['out'].tolist())

    def test_onnxrt_tfidf_vectorizer_empty(self):
        inputi = numpy.array(
            [[1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]]).astype(numpy.int64)
        output = numpy.array([[1., 1., 1.]]).astype(numpy.float32)

        ngram_counts = numpy.array([0, 0]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2]).astype(numpy.int64)
        pool_int64s = numpy.array([  # unigrams
            5, 6, 7, 8, 6, 7]).astype(numpy.int64)   # bigrams

        op = OnnxTfIdfVectorizer(
            'tokens', op_version=get_opset_number_from_onnx(),
            mode='TF', min_gram_length=2, max_gram_length=2,
            max_skip_count=0, ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes, pool_int64s=pool_int64s,
            output_names=['out'])
        onx = op.to_onnx(inputs=[('tokens', Int64TensorType())],
                         outputs=[('out', FloatTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'tokens': inputi})
        self.assertEqual(output.tolist(), res['out'].tolist())

    @ignore_warnings(UserWarning)
    def test_onnxrt_python_count_vectorizer(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?'])
        vect = CountVectorizer()
        vect.fit(corpus)
        exp = vect.transform(corpus)
        onx = to_onnx(vect, corpus, target_opset=get_opset_number_from_onnx())
        oinf = OnnxInference(onx)
        got = oinf.run({'X': corpus})
        self.assertEqualArray(exp.todense(), got['variable'])


if __name__ == "__main__":
    unittest.main()
