"""
@file
@brief Wrapper around :epkg:`onnxruntime`.

.. versionadded:: 0.6
"""
import os
from onnx import numpy_helper

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


def prepare_c_profiling(model_onnx, inputs, dest=None):
    """
    Prepares model and data to be profiled with tool `perftest
    <https://github.com/microsoft/onnxruntime/tree/
    master/onnxruntime/test/perftest>`_ (onnxruntime) or
    `onnx_test_runner <https://github.com/microsoft/
    onnxruntime/blob/master/docs/Model_Test.md>`_.
    It saves the model in folder
    *dest* and dumps the inputs in a subfolder.

    :param model_onnx: onnx model
    :param inputs: inputs as a list of a dictionary
    :param dest: destination folder, None means the current folder
    :return: command line to use
    """
    if dest is None:
        dest = "."
    if not os.path.exists(dest):
        os.makedirs(dest)
    dest = os.path.abspath(dest)
    name = "model.onnx"
    model_bytes = model_onnx.SerializeToString()
    with open(os.path.join(dest, name), "wb") as f:
        f.write(model_bytes)
    sess = InferenceSession(model_bytes)
    input_names = [_.name for _ in sess.get_inputs()]
    if isinstance(inputs, list):
        dict_inputs = dict(zip(input_names, inputs))
    else:
        dict_inputs = inputs
        inputs = [dict_inputs[n] for n in input_names]
    outputs = sess.run(None, dict_inputs)
    sub = os.path.join(dest, "test_data_set_0")
    if not os.path.exists(sub):
        os.makedirs(sub)
    for i, v in enumerate(inputs):
        n = os.path.join(sub, "input_%d.pb" % i)
        pr = numpy_helper.from_array(v)
        with open(n, "wb") as f:
            f.write(pr.SerializeToString())
    for i, v in enumerate(outputs):
        n = os.path.join(sub, "output_%d.pb" % i)
        pr = numpy_helper.from_array(v)
        with open(n, "wb") as f:
            f.write(pr.SerializeToString())

    cmd = 'onnx_test_runner -e cpu -r 100 -c 1 "%s"' % dest
    return cmd
