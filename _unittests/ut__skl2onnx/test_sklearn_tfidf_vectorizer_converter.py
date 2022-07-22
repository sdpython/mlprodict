"""
@brief      test tfidf (time=8s)
"""
import unittest
import copy
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from skl2onnx.common.data_types import StringTensorType, FloatTensorType
from mlprodict.onnx_conv import to_onnx
from mlprodict.testing.test_utils import dump_data_and_model
from mlprodict.tools.ort_wrapper import InferenceSession
from mlprodict import __max_supported_opset__ as TARGET_OPSET


class TestSklearnTfidfVectorizer(ExtTestCase):

    def get_options(self):
        return {TfidfVectorizer: {"tokenexp": None}}

    def test_model_tfidf_vectorizer11(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11-OneOff-SklCol")

        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': corpus})[0]
        self.assertEqual(res.shape, (4, 9))

    def test_model_tfidf_vectorizer11_nolowercase(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None, lowercase=False)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11NoL-OneOff-SklCol")

        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': corpus})[0]
        self.assertEqual(res.shape, (4, 11))

    def test_model_tfidf_vectorizer11_compose(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        corpus = numpy.hstack([corpus, corpus])
        y = numpy.array([0, 1, 0, 1])
        model = ColumnTransformer([
            ('a', TfidfVectorizer(), 0),
            ('b', TfidfVectorizer(), 1)])
        model.fit(corpus, y)
        model_onnx = to_onnx(
            model, initial_types=[("input", StringTensorType([None, 2]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': corpus})[0]
        exp = model.transform(corpus)
        self.assertEqualArray(res, exp)

    def test_model_tfidf_vectorizer11_empty_string_case1(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            ' ',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus[:3].ravel())
        model_onnx = to_onnx(
            vect, initial_types=[('input', StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)

        # TfidfVectorizer in onnxruntime fails with empty strings,
        # which was fixed in version 0.3.0 afterward
        dump_data_and_model(
            corpus[2:], vect, model_onnx,
            basename="SklearnTfidfVectorizer11EmptyStringSepCase1-"
                     "OneOff-SklCol")

    def test_model_tfidf_vectorizer11_empty_string_case2(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        # onnxruntime fails with empty strings
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11EmptyString-OneOff-SklCol")

    def test_model_tfidf_vectorizer11_out_vocabulary(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        corpus = numpy.array([
            "AZZ ZZ This is the first document.",
            "BZZ ZZ This document is the second document.",
            "ZZZ ZZ And this is the third one.",
            "WZZ ZZ Is this the first document?",
        ]).reshape((4, 1))
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11OutVocab-OneOff-SklCol")

    def test_model_tfidf_vectorizer22(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(2, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer22-OneOff-SklCol")

    def test_model_tfidf_vectorizer21(self):
        corpus = numpy.array(["AA AA", "AA AA BB"]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer22S-OneOff-SklCol")

    def test_model_tfidf_vectorizer12(self):
        corpus = numpy.array([
            "first document.",
            "third one.",
        ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(vect, initial_types=
                                     [("input", StringTensorType([None, 1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer22-OneOff-SklCol")

    def test_model_tfidf_vectorizer12_normL1(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm="l1")
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer22L1-OneOff-SklCol")

    def test_model_tfidf_vectorizer12_normL2(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm="l2")
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer22L2-OneOff-SklCol")

    def test_model_tfidf_vectorizer13(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 3), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer13-OneOff-SklCol")

    @unittest.skipIf(True, reason="Discrepancies due to special characters.")
    def test_model_tfidf_vectorizer11parenthesis_class(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the (first) document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        extra = {
            TfidfVectorizer: {
                "separators": [
                    " ", "\\.", "\\?", ",", ";", ":", "\\!", "\\(", "\\)"]}}
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=extra, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        # This test depends on this issue:
        # https://github.com/Microsoft/onnxruntime/issues/957.
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisClass-OneOff-SklCol")

    @unittest.skipIf(True, reason="Discrepancies due to special characters.")
    def test_model_tfidf_vectorizer11_idparenthesis_id(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the (first) document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())

        extra = {
            id(vect): {
                "sep2": [" ", ".", "?", ",", ";", ":", "!", "(", ")"]}}
        try:
            to_onnx(
                vect, initial_types=[("input", StringTensorType([None, 1]))],
                options=extra, target_opset=TARGET_OPSET)
        except (RuntimeError, NameError):
            pass

        extra = {
            id(vect): {
                "separators": [
                    " ", "[.]", "\\?", ",", ";", ":", "\\!", "\\(", "\\)"]}}
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=extra, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        # This test depends on this issue:
        # https://github.com/Microsoft/onnxruntime/issues/957.
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisId-OneOff-SklCol")

    def test_model_tfidf_vectorizer_binary(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(binary=True)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(vect, initial_types=
                                     [("input", StringTensorType([None, 1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizerBinary-OneOff-SklCol")

    def test_model_tfidf_vectorizer11_64(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer1164-OneOff-SklCol")

        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': corpus})[0]
        self.assertEqual(res.shape, (4, 9))

    def test_tfidf_svm(self):
        data = [
            ["schedule a meeting", 0],
            ["schedule a sync with the team", 0],
            ["slot in a meeting", 0],
            ["call ron", 1],
            ["make a phone call", 1],
            ["call in on the phone", 2]]
        docs = [doc for (doc, _) in data]
        labels = [label for (_, label) in data]

        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(docs)
        embeddings = vectorizer.transform(docs)
        dim = embeddings.shape[1]

        clf = SVC()
        clf.fit(embeddings, labels)
        embeddings = embeddings.astype(numpy.float32).todense()
        exp = clf.predict(embeddings)

        initial_type = [('input', FloatTensorType([None, dim]))]
        model_onnx = to_onnx(
            clf, initial_types=initial_type, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': embeddings})[0]
        self.assertEqualArray(exp, res)

    def test_model_tfidf_vectorizer_nan(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        options = copy.deepcopy(self.get_options())
        options[TfidfVectorizer]['nan'] = True
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=options, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': corpus})[0]
        self.assertEqual(res.shape, (4, 9))
        self.assertTrue(numpy.isnan(res[0, 0]))

    def test_model_tfidf_vectorizer11_custom_vocabulary(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vc = ["first", "second", "third", "document", "this"]
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None, vocabulary=vc)
        vect.fit(corpus.ravel())
        self.assertFalse(hasattr(vect, "stop_words_"))
        model_onnx = to_onnx(
            vect, initial_types=[("input", StringTensorType([None, 1]))],
            options=self.get_options(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11CustomVocab-OneOff-SklCol")


if __name__ == "__main__":
    # TestSklearnTfidfVectorizer().test_model_tfidf_vectorizer11_out_vocabulary()
    unittest.main(verbosity=2)
