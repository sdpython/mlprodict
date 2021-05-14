"""
@file
@brief Wrapper around :epkg:`onnxruntime`.

.. versionadded:: 0.6
"""
try:
    from onnxruntime import (  # pylint: disable=W0611
        SessionOptions, RunOptions,
        InferenceSession as OrtInferenceSession,
        __version__ as onnxrt_version,
        GraphOptimizationLevel)
except ImportError:
    SessionOptions = None
    RunOptions = None
    OrtInferenceSession = None
    onnxrt_version = "0.0.0"
    GraphOptimizationLevel = None

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=W0611
        Fail as OrtFail,
        NotImplemented as OrtNotImplemented,
        InvalidArgument as OrtInvalidArgument,
        InvalidGraph as OrtInvalidGraph,
        RuntimeException as OrtRuntimeException)
except ImportError:
    SessionOptions = None
    RunOptions = None
    InferenceSession = None
    onnxrt_version = "0.0.0"
    GraphOptimizationLevel = None
    OrtFail = RuntimeError
    OrtNotImplemented = RuntimeError
    OrtInvalidArgument = RuntimeError
    OrtInvalidGraph = RuntimeError
    OrtRuntimeException = RuntimeError


class InferenceSession:  # pylint: disable=E0102
    """
    Wrappers around InferenceSession from :epkg:`onnxruntime`.

    :param onnx_bytes: onnx bytes
    :param session_options: session options
    """

    def __init__(self, onnx_bytes, sess_options=None, log_severity_level=4):
        if InferenceSession is None:
            raise ImportError(  # pragma: no cover
                "onnxruntime is not available.")
        self.log_severity_level = log_severity_level
        if sess_options is None:
            self.so = SessionOptions()
            self.so.log_severity_level = log_severity_level
            self.sess = OrtInferenceSession(onnx_bytes, sess_options=self.so)
        else:
            self.sess = OrtInferenceSession(
                onnx_bytes, sess_options=sess_options)
        self.ro = RunOptions()
        self.ro.log_severity_level = log_severity_level

    def run(self, output_names, input_feed, run_options=None):
        """
        Executes the ONNX graph.

        :param output_names: None for all, a name for a specific output
        :param input_feed: dictionary of inputs
        :param run_options: None or RunOptions
        :return: array
        """
        return self.sess.run(output_names, input_feed, run_options or self.ro)

    def get_inputs(self):
        "Returns input types."
        return self.sess.get_inputs()

    def get_outputs(self):
        "Returns output types."
        return self.sess.get_outputs()

    def end_profiling(self):
        "Ends profiling."
        return self.sess.end_profiling()
