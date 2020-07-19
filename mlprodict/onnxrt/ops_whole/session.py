# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_whole*.
"""
from io import BytesIO
import onnx
from onnxruntime import InferenceSession, SessionOptions, RunOptions
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        Fail as OrtFail,
        InvalidGraph as OrtInvalidGraph,
        InvalidArgument as OrtInvalidArgument,
        NotImplemented as OrtNotImplemented,
        RuntimeException as OrtRuntimeException,
    )
except ImportError:  # pragma: no cover
    OrtFail = Exception
    OrtNotImplemented = RuntimeError
    OrtInvalidGraph = RuntimeError
    OrtInvalidArgument = RuntimeError
    OrtRuntimeException = RuntimeError
from ...tools.asv_options_helper import display_onnx


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
            raise NotImplementedError(  # pragma: no cover
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
        try:
            self.sess = InferenceSession(onnx_data, sess_options=sess_options)
        except (OrtFail, OrtNotImplemented, OrtInvalidGraph,
                OrtInvalidArgument, OrtRuntimeException, RuntimeError) as e:
            raise RuntimeError(
                "Unable to create InferenceSession due to '{}'\n{}.".format(
                    e, display_onnx(onnx.load(BytesIO(onnx_data))))) from e

    def run(self, inputs):
        """
        Computes the predictions.

        @param      inputs      dictionary *{variable, value}*
        @return                 list of outputs
        """
        return self.sess.run(None, inputs, self.run_options)
