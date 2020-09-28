"""
@file
@brief Inspired from :epkg:`sklearn-onnx`, handles two backends.
"""
from .utils_backend_onnxruntime import compare_runtime as compare_runtime_ort
from .utils_backend_python import compare_runtime as compare_runtime_pyrt


def compare_backend(backend, test, decimal=5, options=None, verbose=False,
                    context=None, comparable_outputs=None,
                    intermediate_steps=False, classes=None,
                    disable_optimisation=False):
    """
    The function compares the expected output (computed with
    the model before being converted to ONNX) and the ONNX output.

    :param backend: backend to use to run the comparison
    :param test: dictionary with the following keys:
        - *onnx*: onnx model (filename or object)
        - *expected*: expected output (filename pkl or object)
        - *data*: input data (filename pkl or object)
    :param decimal: precision of the comparison
    :param options: comparison options
    :param context: specifies custom operators
    :param comparable_outputs: compare only these outputs
    :param verbose: in case of error, the function may print
        more information on the standard output
    :param intermediate_steps: displays intermediate steps
        in case of an error
    :param classes: classes names (if option 'nocl' is used)
    :param disable_optimisation: disable optimisation onnxruntime
        could do

    The function does not return anything but raises an error
    if the comparison failed.
    :return: tuple (output, lambda function to call onnx predictions)
    """
    if backend == "onnxruntime":
        return compare_runtime_ort(
            test, decimal, options=options, verbose=verbose,
            comparable_outputs=comparable_outputs,
            intermediate_steps=False, classes=classes,
            disable_optimisation=disable_optimisation)
    if backend == "python":
        return compare_runtime_pyrt(
            test, decimal, options=options, verbose=verbose,
            comparable_outputs=comparable_outputs,
            intermediate_steps=intermediate_steps, classes=classes,
            disable_optimisation=disable_optimisation)
    raise ValueError(  # pragma: no cover
        "Does not support backend '{0}'.".format(backend))
