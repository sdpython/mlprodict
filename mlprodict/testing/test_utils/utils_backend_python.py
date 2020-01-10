"""
@file
@brief Inspired from skl2onnx, handles two backends.
"""
from ...onnxrt import OnnxInference
from .utils_backend_common_compare import compare_runtime_session


class MockVariableName:
    "A string."

    def __init__(self, name):
        self.name = name


class OnnxInference2(OnnxInference):
    "onnxruntime API"

    def run(self, name, inputs, *args, **kwargs):  # pylint: disable=W0221
        "onnxruntime API"
        res = OnnxInference.run(self, inputs, **kwargs)
        if name is None:
            return [res[n] for n in self.output_names]
        if name in res:
            return res[name]
        raise RuntimeError("Unable to find output '{}'.".format(name))

    def get_inputs(self):
        "onnxruntime API"
        return [MockVariableName(n) for n in self.input_names]

    def get_outputs(self):
        "onnxruntime API"
        return [MockVariableName(n) for n in self.output_names]


def compare_runtime(test, decimal=5, options=None,
                    verbose=False, context=None, comparable_outputs=None,
                    intermediate_steps=False, classes=None):
    """
    The function compares the expected output (computed with
    the model before being converted to ONNX) and the ONNX output
    produced with module *onnxruntime*.

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
    :return: tuple (outut, lambda function to run the predictions)

    The function does not return anything but raises an error
    if the comparison failed.
    """
    return compare_runtime_session(
        OnnxInference2, test, decimal=decimal, options=options,
        verbose=verbose, context=context,
        comparable_outputs=comparable_outputs,
        intermediate_steps=intermediate_steps,
        classes=classes)
