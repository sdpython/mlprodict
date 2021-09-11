# coding: utf-8
"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper.version_helper import compare_module_version
from skl2onnx import __version__ as sk2ver
from skl2onnx.common.data_types import (
    StringTensorType, FloatTensorType, Int64TensorType)
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxStringNormalizer, OnnxTfIdfVectorizer, OnnxLabelEncoder)
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnx_conv.onnx_ops import OnnxTokenizer
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools import get_opset_number_from_onnx


class TestOnnxrtPythonRuntimeMlText(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxrt_label_encoder_strings(self):

        corpus = numpy.array(['AA', 'BB', 'AA', 'CC'])
        op = OnnxLabelEncoder(
            'text', op_version=get_opset_number_from_onnx(),
            keys_strings=['AA', 'BB', 'CC'],
            values_strings=['LEAA', 'LEBB', 'LECC'],
            output_names=['out'])
        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqual(list(res['out']), ['LEAA', 'LEBB', 'LEAA', 'LECC'])

    def test_onnxrt_label_encoder_floats(self):

        corpus = numpy.array([0.1, 0.2, 0.3, 0.2], dtype=numpy.float32)
        op = OnnxLabelEncoder(
            'text', op_version=get_opset_number_from_onnx(),
            keys_floats=[0.1, 0.2, 0.3],
            values_floats=[0.3, 0.4, 0.5],
            output_names=['out'])
        onx = op.to_onnx(inputs=[('text', FloatTensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'text': corpus})
        self.assertEqualArray(
            res['out'], numpy.array([0.3, 0.4, 0.5, 0.4], dtype=numpy.float32))

    def test_onnxrt_label_encoder_raise(self):

        self.assertRaise(
            lambda: OnnxLabelEncoder(
                'text', op_version=get_opset_number_from_onnx(),
                keys_strings=['AA', 'BB', 'CC'],
                classes_strings=['LEAA', 'LEBB', 'LECC'],
                output_names=['out']),
            TypeError)

        op = OnnxLabelEncoder(
            'text', op_version=get_opset_number_from_onnx(),
            keys_strings=['AA', 'BB', 'CC'],
            values_floats=[0.1, 0.2, 0.3],
            output_names=['out'])

        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        self.assertRaise(lambda: OnnxInference(onx), RuntimeError)

        op = OnnxLabelEncoder(
            'text', op_version=get_opset_number_from_onnx(),
            keys_strings=['AA', 'BB', 'CC'],
            values_strings=[],
            output_names=['out'])

        onx = op.to_onnx(inputs=[('text', StringTensorType())])
        self.assertRaise(lambda: OnnxInference(onx), RuntimeError)

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

    @unittest.skipIf(compare_module_version(sk2ver, '1.9.3') < 0,
                     reason="fails on that example")
    @ignore_warnings(UserWarning)
    def test_multi_output_classifier(self):
        dfx = pandas.DataFrame(
            {'CAT1': ['985332', '985333', '985334', '985335', '985336'],
             'CAT2': ['1985332', '1985333', '1985334', '1985335', '1985336'],
             'TEXT': ["abc abc", "abc def", "def ghj", "abcdef", "abc ii"]})
        dfy = pandas.DataFrame(
            {'REAL': [5, 6, 7, 6, 5],
             'CATY': [0, 1, 0, 1, 0]})

        cat_features = ['CAT1', 'CAT2']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        textual_feature = 'TEXT'
        count_vect_transformer = Pipeline(steps=[
            ('count_vect', CountVectorizer(
                max_df=0.8, min_df=0.05, max_features=1000))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat_transform', categorical_transformer, cat_features),
                ('count_vector', count_vect_transformer, textual_feature)])
        model_RF = RandomForestClassifier(random_state=42, max_depth=50)
        rf_clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MultiOutputClassifier(estimator=model_RF))])
        rf_clf.fit(dfx, dfy)
        expected_label = rf_clf.predict(dfx)
        expected_proba = rf_clf.predict_proba(dfx)

        inputs = {'CAT1': dfx['CAT1'].values.reshape((-1, 1)),
                  'CAT2': dfx['CAT2'].values.reshape((-1, 1)),
                  'TEXT': dfx['TEXT'].values.reshape((-1, 1))}
        onx = to_onnx(rf_clf, dfx, target_opset=get_opset_number_from_onnx())
        sess = OnnxInference(onx)

        got = sess.run(inputs)
        self.assertArrayEqual(expected_label, got[0])
        self.assertEqual(len(expected_proba), len(got[1]))
        for e, g in zip(expected_proba, got[1]):
            self.assertArrayEqual(e, g, decimal=5)


if __name__ == "__main__":
    unittest.main()
