"""
@brief      test log(time=4s)
"""
import unittest
import io
import pickle
import base64
import os
import numpy
from pyquickhelper.pycode import ExtTestCase
try:
    from mlprodict.sklapi.onnx_tokenizer import (
        SentencePieceTokenizerTransformer, GPT2TokenizerTransformer)
except ImportError:
    GPT2TokenizerTransformer = None


class TestOnnxTokenizer(ExtTestCase):

    def _load_piece(self):
        fullname = os.path.join(
            os.path.dirname(__file__), "data",
            "test_sentencepiece_ops_model__6.txt")
        with open(fullname, "r") as f:
            content = f.read()
        t = base64.decodebytes(content.encode())
        b64 = base64.b64encode(t)
        return numpy.array(list(t), dtype=numpy.uint8), b64

    @unittest.skipIf(GPT2TokenizerTransformer is None,
                     reason="onnxruntime-extensions not available")
    def test_sentence_piece_tokenizer_transformer(self):
        model, model_b64 = self._load_piece()
        cints = bytes(model.tolist())
        cb64 = base64.b64encode(cints)
        self.assertEqual(model_b64, cb64)

        for alpha in [0, 0.5]:
            for nbest_size in [0, 1]:
                for bools in range(0, 8):
                    params = dict(
                        model=model_b64,
                        nbest_size=numpy.array(
                            [nbest_size], dtype=numpy.int64),
                        alpha=numpy.array([alpha], dtype=numpy.float32),
                        add_bos=numpy.array([bools & 1], dtype=numpy.bool_),
                        add_eos=numpy.array([bools & 2], dtype=numpy.bool_),
                        reverse=numpy.array([bools & 4], dtype=numpy.bool_))
                    model = SentencePieceTokenizerTransformer(**params)
                    inputs = numpy.array(
                        ["Hello world", "Hello world louder"],
                        dtype=numpy.object)
                    model.fit(inputs)
                    got = model.transform(inputs)
                    self.assertEqual(got.shape, (2, 21870))

                    buf = io.BytesIO()
                    pickle.dump(model, buf)
                    buf2 = io.BytesIO(buf.getvalue())
                    restored = pickle.load(buf2)
                    got2 = restored.transform(inputs)
                    self.assertEqualArray(got.todense(), got2.todense())
                    return

    @unittest.skipIf(GPT2TokenizerTransformer is None,
                     reason="onnxruntime-extensions not available")
    def test_gpt2_tokenizer_transformer(self):
        vocab = os.path.join(
            os.path.dirname(__file__), "data", "gpt2.vocab")
        merges = os.path.join(
            os.path.dirname(__file__), "data", "gpt2.merges.txt")
        with open(vocab, "rb") as f:
            vocab_content = f.read()
        with open(merges, "rb") as f:
            merges_content = f.read()

        model = GPT2TokenizerTransformer(
            vocab=vocab_content, merges=merges_content)
        inputs = numpy.array(
            ["Hello world", "Hello world louder"],
            dtype=numpy.object)
        model.fit(inputs)
        got = model.transform(inputs)
        self.assertEqual(got.shape, (2, 27090))

        buf = io.BytesIO()
        pickle.dump(model, buf)
        buf2 = io.BytesIO(buf.getvalue())
        restored = pickle.load(buf2)
        got2 = restored.transform(inputs)
        self.assertEqualArray(got.todense(), got2.todense())


if __name__ == '__main__':
    unittest.main()
