"""
@file
@brief
"""
from io import BytesIO
from onnx import load, load_model
# from onnx import helper, shape_inference


class OnnxInference:
    """
    Loads an :epkg:`ONNX` file or object or stream.
    Computes the output of the :epkg:`ONNX` graph.
    """

    def __init__(self, onnx_or_bytes_or_stream):
        """
        @param      onnx_or_bytes_or_stream     :epkg:`onnx` object,
                                                bytes, or filename or stream
        """
        if isinstance(onnx_or_bytes_or_stream, bytes):
            self.obj = load_model(BytesIO(onnx_or_bytes_or_stream))
        elif isinstance(onnx_or_bytes_or_stream, BytesIO):
            self.obj = load_model(onnx_or_bytes_or_stream)
        elif isinstance(onnx_or_bytes_or_stream, str):
            self.obj = load(onnx_or_bytes_or_stream)
        elif hasattr(onnx_or_bytes_or_stream, 'graph'):
            self.obj = onnx_or_bytes_or_stream
        else:
            raise TypeError("Unable to handle type {}.".format(
                type(onnx_or_bytes_or_stream)))

    def __str__(self):
        """
        usual
        """
        return str(self.obj)
