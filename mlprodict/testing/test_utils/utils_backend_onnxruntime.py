"""
@file
@brief Inspired from skl2onnx, handles two backends.
"""
from pyquickhelper.pycode import is_travis_or_appveyor
from .utils_backend_common_compare import compare_runtime_session
from ...tools.ort_wrapper import (
    InferenceSession, GraphOptimizationLevel, SessionOptions)


def _capture_output(fct, kind):
    if is_travis_or_appveyor():
        return fct(), None, None  # pragma: no cover
    try:
        from cpyquickhelper.io import capture_output
    except ImportError:
        # cpyquickhelper not available
        return fct(), None, None  # pragma: no cover
    return capture_output(fct, kind)  # pragma: no cover


class InferenceSession2:
    """
    Overwrites class *InferenceSession* to capture
    the standard output and error.
    """

    def __init__(self, *args, **kwargs):
        "Overwrites the constructor."
        runtime_options = kwargs.pop('runtime_options', {})
        disable_optimisation = runtime_options.pop(
            'disable_optimisation', False)
        if disable_optimisation:
            if 'sess_options' in kwargs:
                raise RuntimeError(  # pragma: no cover
                    "Incompatible options, 'disable_options' and 'sess_options' cannot "
                    "be sepcified at the same time.")
            kwargs['sess_options'] = SessionOptions()
            kwargs['sess_options'].graph_optimization_level = (
                GraphOptimizationLevel.ORT_DISABLE_ALL)
        self.sess, self.outi, self.erri = _capture_output(
            lambda: InferenceSession(*args, **kwargs), 'c')

    def run(self, *args, **kwargs):
        "Overwrites method *run*."
        res, self.outr, self.errr = _capture_output(
            lambda: self.sess.run(*args, **kwargs), 'c')
        return res

    def get_inputs(self, *args, **kwargs):
        "Overwrites method *get_inputs*."
        return self.sess.get_inputs(*args, **kwargs)

    def get_outputs(self, *args, **kwargs):
        "Overwrites method *get_outputs*."
        return self.sess.get_outputs(*args, **kwargs)


def compare_runtime(test, decimal=5, options=None,
                    verbose=False, context=None, comparable_outputs=None,
                    intermediate_steps=False, classes=None,
                    disable_optimisation=False):
    """
    The function compares the expected output (computed with
    the model before being converted to ONNX) and the ONNX output
    produced with module :epkg:`onnxruntime` or :epkg:`mlprodict`.

    :param test: dictionary with the following keys:
        - *onnx*: onnx model (filename or object)
        - *expected*: expected output (filename pkl or object)
        - *data*: input data (filename pkl or object)
    :param decimal: precision of the comparison
    :param options: comparison options
    :param context: specifies custom operators
    :param verbose: in case of error, the function may print
        more information on the standard output
    :param comparable_outputs: compare only these outputs
    :param intermediate_steps: displays intermediate steps
        in case of an error
    :param classes: classes names (if option 'nocl' is used)
    :param disable_optimisation: disable optimisation onnxruntime
        could do
    :return: tuple (outut, lambda function to run the predictions)

    The function does not return anything but raises an error
    if the comparison failed.
    """
    return compare_runtime_session(
        InferenceSession2, test, decimal=decimal, options=options,
        verbose=verbose, context=context,
        comparable_outputs=comparable_outputs,
        intermediate_steps=intermediate_steps,
        classes=classes, disable_optimisation=disable_optimisation)
