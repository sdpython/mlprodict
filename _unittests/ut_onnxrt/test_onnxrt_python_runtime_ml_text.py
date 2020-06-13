# coding: utf-8
"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.common.data_types import StringTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxStringNormalizer)
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

    def skip_test_onnxrt_python_count_vectorizer(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?'])
        vect = CountVectorizer()
        vect.fit(corpus)
        exp = vect.transform(corpus)
        onx = to_onnx(vect, corpus, target_opset=get_opset_number_from_onnx())
        print(onx)
        oinf = OnnxInference(onx)
        got = oinf.run({'X': corpus})
        self.assertEqualArray(exp, got['variable'])


if __name__ == "__main__":
    unittest.main()
