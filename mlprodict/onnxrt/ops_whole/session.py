# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_whole*.
"""
from onnxruntime import InferenceSession


class OnnxWholeSession:
    """
    Runs the prediction for a whole :epkg:`ONNX`,
    it lets the runtime handle the graph logic as well.
    """

    def __init__(self, onnx_data, runtime):
        """
        @param      onnx_data       :epkg:`ONNX` model or data
        @param      runtime         runtime to be used,
                                    mostly :epkg:`onnxruntime`
        """
        if runtime != 'onnxruntime-whole':
            raise NotImplementedError(
                "runtime '{}' is not implemented.".format(runtime))
        if hasattr(onnx_data, 'SerializeToString'):
            onnx_data = onnx_data.SerializeToString()
        self.runtime = runtime
        self.sess = InferenceSession(onnx_data)

    def run(self, inputs):
        """
        Computes the predictions.

        @param      inputs      dictionary *{variable, value}*
        @return                 list of outputs
        """
        return self.sess.run(None, inputs)
