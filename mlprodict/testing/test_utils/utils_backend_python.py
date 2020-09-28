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

    @property
    def shape(self):
        "returns shape"
        raise NotImplementedError(  # pragma: no cover
            "No shape for '{}'.".format(self.name))

    @property
    def type(self):
        "returns type"
        raise NotImplementedError(  # pragma: no cover
            "No type for '{}'.".format(self.name))


class MockVariableNameShape(MockVariableName):
    "A string and a shape."

    def __init__(self, name, sh):
        MockVariableName.__init__(self, name)
        self._shape = sh

    @property
    def shape(self):
        "returns shape"
        return self._shape


class MockVariableNameShapeType(MockVariableNameShape):
    "A string and a shape and a type."

    def __init__(self, name, sh, stype):
        MockVariableNameShape.__init__(self, name, sh)
        self._stype = stype

    @property
    def type(self):
        "returns type"
        return self._stype


class OnnxInference2(OnnxInference):
    "onnxruntime API"

    def run(self, name, inputs, *args, **kwargs):  # pylint: disable=W0221
        "onnxruntime API"
        res = OnnxInference.run(self, inputs, **kwargs)
        if name is None:
            return [res[n] for n in self.output_names]
        if name in res:  # pragma: no cover
            return res[name]
        raise RuntimeError(  # pragma: no cover
            "Unable to find output '{}'.".format(name))

    def get_inputs(self):
        "onnxruntime API"
        return [MockVariableNameShapeType(*n) for n in self.input_names_shapes_types]

    def get_outputs(self):
        "onnxruntime API"
        return [MockVariableNameShape(*n) for n in self.output_names_shapes]

    def run_in_scan(self, inputs, verbose=0, fLOG=None):
        "Instance to run in operator scan."
        return OnnxInference.run(self, inputs, verbose=verbose, fLOG=fLOG)


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
    :param disable_optimisation: disable optimisation the runtime may do
    :return: tuple (outut, lambda function to run the predictions)

    The function does not return anything but raises an error
    if the comparison failed.
    """
    return compare_runtime_session(
        OnnxInference2, test, decimal=decimal, options=options,
        verbose=verbose, context=context,
        comparable_outputs=comparable_outputs,
        intermediate_steps=intermediate_steps,
        classes=classes, disable_optimisation=disable_optimisation)
