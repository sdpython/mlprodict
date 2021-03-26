"""
@file
@brief Custom operator Tokenizer.
"""
from skl2onnx.algebra.onnx_operator import OnnxOperator


class OnnxTokenizer_1(OnnxOperator):
    """
    Defines a custom operator not defined by ONNX
    specifications but in onnxruntime.
    """

    since_version = 1
    expected_inputs = [('text', 'T')]
    expected_outputs = [('tokens', 'T')]
    input_range = [1, 1]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'mlprodict'
    operator_name = 'Tokenizer'
    past_version = {}

    def __init__(self, text, mark=0, mincharnum=1,
                 pad_value='#', separators=None,
                 tokenexp='[a-zA-Z0-9_]+', stopwords=None,
                 op_version=None, **kwargs):
        """
        :param text: array or OnnxOperatorMixin
        :param mark: see :epkg:`Tokenizer`
        :param pad_value: see :epkg:`Tokenizer`
        :param separators: see :epkg:`Tokenizer`
        :param tokenexp: see :epkg:`Tokenizer`
        :param stopwords: list of stopwords, addition to :epkg:`Tokenizer`
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        if separators is None:
            separators = []
        if stopwords is None:
            stopwords = []
        OnnxOperator.__init__(
            self, text, mark=mark, mincharnum=mincharnum,
            pad_value=pad_value, separators=separators, tokenexp=tokenexp,
            stopwords=stopwords, op_version=op_version, **kwargs)


OnnxTokenizer = OnnxTokenizer_1
