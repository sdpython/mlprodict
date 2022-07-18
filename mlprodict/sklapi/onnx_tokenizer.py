# pylint: disable=E1101
"""
@file
@brief Wrapper tokenizrs implemented in :epkg:`onnxruntime-extensions`.
"""
from io import BytesIO
import base64
import numpy
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from onnx import helper, TensorProto, load
from onnx.defs import onnx_opset_version
try:
    from onnxruntime_extensions import get_library_path
except ImportError:  # pragma: no cover
    get_library_path = None
from mlprodict import __max_supported_opset__


class TokenizerTransformerBase(BaseEstimator, TransformerMixin):
    """
    Base class for @see cl SentencePieceTokenizerTransformer and
    @see cl GPT2TokenizerTransformer.
    """

    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        from onnxruntime import InferenceSession, SessionOptions  # delayed
        self._InferenceSession = InferenceSession
        self._SessionOptions = SessionOptions

    def __getstate__(self):
        state = BaseEstimator.__getstate__(self)
        del state['sess_']
        del state['_InferenceSession']
        del state['_SessionOptions']
        state['onnx_'] = state['onnx_'].SerializeToString()
        return state

    def __setstate__(self, state):
        if get_library_path is None:
            raise ImportError(  # pragma: no cover
                "onnxruntime_extensions is not installed.")
        from onnxruntime import InferenceSession, SessionOptions  # delayed
        state['onnx_'] = load(BytesIO(state['onnx_']))
        BaseEstimator.__setstate__(self, state)
        self._InferenceSession = InferenceSession
        self._SessionOptions = SessionOptions
        so = SessionOptions()
        so.register_custom_ops_library(get_library_path())
        self.sess_ = InferenceSession(self.onnx_.SerializeToString(), so,
                                      providers=['CPUExecutionProvider'])
        return self


class SentencePieceTokenizerTransformer(TokenizerTransformerBase):
    """
    Wraps `SentencePieceTokenizer
    <https://github.com/microsoft/onnxruntime-extensions/blob/
    main/docs/custom_text_ops.md#sentencepiecetokenizer>`_
    into a :epkg:`scikit-learn` transformer.

    :param model: The sentencepiece model serialized proto as
        stored as a string
    :param nbest_size: tensor(int64) A scalar for sampling.
        `nbest_size = {0,1}`: no sampling is performed.
        (default) `nbest_size > 1`: samples from the nbest_size results.
        `nbest_size < 0`: assuming that nbest_size is infinite and
        samples from the all hypothesis (lattice) using
        forward-filtering-and-backward-sampling algorithm.
    :param alpha: tensor(float) A scalar for a smoothing parameter.
        Inverse temperature for probability rescaling.
    :param reverse: tensor(bool) Reverses the tokenized sequence.
    :param add_bos: tensor(bool) Add beginning of sentence token to the result.
    :param add_eos: tensor(bool) Add end of sentence token to the result
        When reverse=True beginning/end of sentence tokens are added
        after reversing
    :param opset: main opset to use

    Method *fit* produces the following attributes:

    * `onnx_`: onnx graph
    * `sess_`: :epkg:`InferenceSession` used to compute the inference
    """

    def __init__(self, model, nbest_size=1, alpha=0.5, reverse=False,
                 add_bos=False, add_eos=False, opset=None):
        TokenizerTransformerBase.__init__(self)
        if isinstance(model, bytes):
            self.model_b64 = model
        else:
            ints = model.tolist()
            b64 = base64.b64encode(ints)
            self.model_b64 = b64
        self.nbest_size = nbest_size
        self.alpha = alpha
        self.reverse = reverse
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.opset = opset
        if get_library_path is None:
            raise ImportError(  # pragma: no cover
                "onnxruntime_extensions is not installed.")

    def fit(self, X, y=None, sample_weight=None):
        """
        The model is not trains this method is still needed to
        set the instance up and ready to transform.

        :param X: array of strings
        :param y: unused
        :param sample_weight: unused
        :return: self
        """
        self.onnx_ = self._create_model(
            self.model_b64, opset=self.opset)
        so = self._SessionOptions()
        so.register_custom_ops_library(get_library_path())
        self.sess_ = self._InferenceSession(self.onnx_.SerializeToString(), so)
        return self

    @staticmethod
    def _create_model(model_b64, domain='ai.onnx.contrib', opset=None):
        nodes = []
        mkv = helper.make_tensor_value_info
        nodes.append(helper.make_node(
            'SentencepieceTokenizer',
            inputs=['inputs', 'nbest_size', 'alpha', 'add_bos', 'add_eos',
                    'reverse'],
            outputs=['out0', 'out1'],
            model=model_b64,
            name='SentencepieceTokenizeOpName',
            domain='ai.onnx.contrib'))
        inputs = [
            mkv('inputs', TensorProto.STRING, [None]),
            mkv('nbest_size', TensorProto.INT64, [None]),
            mkv('alpha', TensorProto.FLOAT, [None]),
            mkv('add_bos', TensorProto.BOOL, [None]),
            mkv('add_eos', TensorProto.BOOL, [None]),
            mkv('reverse', TensorProto.BOOL, [None])]
        graph = helper.make_graph(
            nodes, 'SentencePieceTokenizerTransformer', inputs, [
                mkv('out0', TensorProto.INT32, [None]),
                mkv('out1', TensorProto.INT64, [None])])
        if opset is None:
            opset = min(__max_supported_opset__, onnx_opset_version())
        model = helper.make_model(graph, opset_imports=[
            helper.make_operatorsetid('', opset)])
        model.opset_import.extend([helper.make_operatorsetid(domain, 1)])
        return model

    def transform(self, X):
        """
        Applies the tokenizers on an array of strings.

        :param X: array to strings.
        :return: sparses matrix with n_features
        """
        out0, out1 = self.sess_.run(['out0', 'out1'],
                                    {'inputs': X, 'nbest_size': self.nbest_size, 'alpha': self.alpha,
                                     'add_bos': self.add_bos, 'add_eos': self.add_eos,
                                     'reverse': self.reverse})
        values = numpy.ones(out0.shape[0], dtype=numpy.float32)
        return csr_matrix((values, out0, out1))


class GPT2TokenizerTransformer(TokenizerTransformerBase):
    """
    Wraps `GPT2Tokenizer
    <https://github.com/microsoft/onnxruntime-extensions/blob/
    main/docs/custom_text_ops.md#gpt2tokenizer>`_
    into a :epkg:`scikit-learn` transformer.

    :param vocab: The content of the vocabulary file,
        its format is same with hugging face.
    :param merges: The content of the merges file,
        its format is same with hugging face.
    :param padding_length: When the input is a set of query,
        the tokenized result is ragged tensor, so we need to pad
        the tensor to tidy tensor and the *padding_length* indicates
        the strategy of the padding.
        When the *padding_length* equals -1, we will pad the tensor
        to length of longest row.
        When the *padding_length* is more than 0, we will pad the tensor
        to the number of padding_length.
    :param opset: main opset to use

    Method *fit* produces the following attributes:

    * `onnx_`: onnx graph
    * `sess_`: :epkg:`InferenceSession` used to compute the inference
    """

    def __init__(self, vocab, merges, padding_length=-1, opset=None):
        TokenizerTransformerBase.__init__(self)
        self.vocab = vocab
        self.merges = merges
        self.padding_length = padding_length
        self.opset = opset
        if get_library_path is None:
            raise ImportError(  # pragma: no cover
                "onnxruntime_extensions is not installed.")

    def fit(self, X, y=None, sample_weight=None):
        """
        The model is not trains this method is still needed to
        set the instance up and ready to transform.

        :param X: array of strings
        :param y: unused
        :param sample_weight: unused
        :return: self
        """
        self.onnx_ = self._create_model(
            self.vocab, self.merges, self.padding_length, opset=self.opset)
        so = self._SessionOptions()
        so.register_custom_ops_library(get_library_path())
        self.sess_ = self._InferenceSession(self.onnx_.SerializeToString(), so)
        return self

    @staticmethod
    def _create_model(vocab, merges, padding_length,
                      domain='ai.onnx.contrib', opset=None):
        nodes = []
        mkv = helper.make_tensor_value_info
        nodes.append(helper.make_node(
            'GPT2Tokenizer',
            inputs=['inputs'],
            outputs=['input_ids', 'attention_mask'],
            vocab=vocab, merges=merges,
            padding_length=padding_length,
            name='GPT2TokenizerName',
            domain='ai.onnx.contrib'))
        inputs = [mkv('inputs', TensorProto.STRING, [None])]
        graph = helper.make_graph(
            nodes, 'GPT2TokenizerTransformer', inputs, [
                mkv('input_ids', TensorProto.INT64, [None, None]),
                mkv('attention_mask', TensorProto.INT64, [None, None])])
        if opset is None:
            opset = min(__max_supported_opset__, onnx_opset_version())
        model = helper.make_model(
            graph, opset_imports=[helper.make_operatorsetid('', opset)])
        model.opset_import.extend([helper.make_operatorsetid(domain, 1)])
        return model

    def transform(self, X):
        """
        Applies the tokenizers on an array of strings.

        :param X: array to strings.
        :return: sparses matrix with n_features
        """
        input_ids, _ = self.sess_.run(
            ['input_ids', 'attention_mask'], {'inputs': X})
        idx = input_ids.ravel()
        values = numpy.ones(idx.shape[0], dtype=numpy.float32)
        rg = numpy.arange(input_ids.shape[0] + 1).astype(numpy.int64)
        rows = rg * input_ids.shape[1]
        return csr_matrix((values, idx, rows))
