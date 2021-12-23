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
        GraphOptimizationLevel,
        set_default_logger_severity)
    from .onnx_inference_ort_helper import get_ort_device, device_to_providers
except ImportError:  # pragma: no cover
    SessionOptions = None
    RunOptions = None
    OrtInferenceSession = None
    onnxrt_version = "0.0.0"
    GraphOptimizationLevel = None
    get_ort_device = None
    device_to_providers = None
    set_default_logger_severity = None

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=W0611
        Fail as OrtFail,
        NotImplemented as OrtNotImplemented,
        InvalidArgument as OrtInvalidArgument,
        InvalidGraph as OrtInvalidGraph,
        RuntimeException as OrtRuntimeException,
        OrtValue as C_OrtValue)
except ImportError:  # pragma: no cover
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
    C_OrtValue = None


class InferenceSession:  # pylint: disable=E0102
    """
    Wrappers around InferenceSession from :epkg:`onnxruntime`.

    :param onnx_bytes: onnx bytes
    :param session_options: session options
    :param log_severity_level: change the logging level
    :param device: device, a string `cpu`, `cuda`, `cuda:0`...
    """

    def __init__(self, onnx_bytes, sess_options=None, log_severity_level=4,
                 device=None):
        if InferenceSession is None:
            raise ImportError(  # pragma: no cover
                "onnxruntime is not available.")
        self.log_severity_level = log_severity_level
        if device is None:
            self.device = get_ort_device('cpu')
        else:
            self.device = get_ort_device(device)
        self.providers = device_to_providers(self.device)
        set_default_logger_severity(3)
        if sess_options is None:
            self.so = SessionOptions()
            self.so.log_severity_level = log_severity_level
            self.sess = OrtInferenceSession(
                onnx_bytes, sess_options=self.so,
                providers=self.providers)
        else:
            self.so = sess_options
            self.sess = OrtInferenceSession(
                onnx_bytes, sess_options=sess_options,
                providers=self.providers)
        self.ro = RunOptions()
        self.ro.log_severity_level = log_severity_level
        self.ro.log_verbosity_level = log_severity_level
        self.output_names = [o.name for o in self.get_outputs()]

    def run(self, output_names, input_feed, run_options=None):
        """
        Executes the ONNX graph.

        :param output_names: None for all, a name for a specific output
        :param input_feed: dictionary of inputs
        :param run_options: None or RunOptions
        :return: array
        """
        if any(map(lambda v: isinstance(v, C_OrtValue),
                   input_feed.values())):
            return self.sess._sess.run_with_ort_values(
                input_feed, self.output_names, run_options or self.ro)
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
        os.makedirs(dest)  # pragma: no cover
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
