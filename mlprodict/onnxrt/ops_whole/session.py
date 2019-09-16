# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_whole*.
"""
from onnxruntime import InferenceSession, SessionOptions, RunOptions


class OnnxWholeSession:
    """
    Runs the prediction for a single :epkg:`ONNX`,
    it lets the runtime handle the graph logic as well.
    """

    def __init__(self, onnx_data, runtime):
        """
        @param      onnx_data       :epkg:`ONNX` model or data
        @param      runtime         runtime to be used,
                                    mostly :epkg:`onnxruntime`
        """
        if runtime != 'onnxruntime1':
            raise NotImplementedError(
                "runtime '{}' is not implemented.".format(runtime))
        if hasattr(onnx_data, 'SerializeToString'):
            onnx_data = onnx_data.SerializeToString()
        self.runtime = runtime
        sess_options = SessionOptions()
        self.run_options = RunOptions()
        try:
            sess_options.sessions_log_verbosity_level = 0
        except AttributeError:  # pragma: no cover
            # onnxruntime not recent enough.
            pass
        try:
            self.run_options.run_log_verbosity_level = 0
        except AttributeError:  # pragma: no cover
            # onnxruntime not recent enough.
            pass
        self.sess = InferenceSession(onnx_data, sess_options=sess_options)

    def run(self, inputs):
        """
        Computes the predictions.

        @param      inputs      dictionary *{variable, value}*
        @return                 list of outputs
        """
        return self.sess.run(None, inputs, self.run_options)
