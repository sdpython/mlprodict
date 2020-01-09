"""
@file
@brief Inspired from skl2onnx, handles two backends.
"""
import numpy
import onnx
import pandas
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        InvalidArgument as OrtInvalidArgument
    )
except ImportError:
    OrtInvalidArgument = RuntimeError
from ...onnxrt import OnnxInference
from .utils_backend_common import (
    load_data_and_model,
    extract_options,
    ExpectedAssertionError,
    OnnxBackendAssertionError,
    OnnxRuntimeMissingNewOnnxOperatorException,
    _compare_expected,
    _create_column,
)


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


def compare_runtime(test, decimal=5, options=None,  # pylint: disable=R0912
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
    lambda_onnx = None
    if context is None:
        context = {}
    load = load_data_and_model(test, **context)
    if verbose:  # pragma no cover
        print("[compare_runtime] test '{}' loaded".format(test['onnx']))

    onx = test['onnx']
    if options is None:
        if isinstance(onx, str):
            options = extract_options(onx)
        else:
            options = {}
    elif options is None:
        options = {}
    elif not isinstance(options, dict):
        raise TypeError("options must be a dictionary.")

    if verbose:  # pragma no cover
        print("[compare_runtime] InferenceSession('{}')".format(onx))

    try:
        sess = OnnxInference2(onx)
    except ExpectedAssertionError as expe:  # pragma no cover
        raise expe
    except Exception as e:  # pylint: disable=W0703
        if "CannotLoad" in options:  # pragma no cover
            raise ExpectedAssertionError(
                "Unable to load onnx '{0}' due to\n{1}".format(onx, e))
        else:  # pragma no cover
            if intermediate_steps:
                raise NotImplementedError(
                    "intermediate steps are not implemented for this backend.")
            if verbose:  # pragma no cover
                model = onnx.load(onx)
                smodel = "\nJSON ONNX\n" + str(model)
            else:
                smodel = ""
            if ("NOT_IMPLEMENTED : Could not find an implementation "
                    "for the node" in str(e)):
                # onnxruntime does not implement a specific node yet.
                raise OnnxRuntimeMissingNewOnnxOperatorException(
                    "onnxruntime does not implement a new operator "
                    "'{0}'\n{1}\nONNX\n{2}".format(
                        onx, e, smodel))
            raise OnnxBackendAssertionError(
                "Unable to load onnx '{0}'\nONNX\n{1}\n{2}".format(
                    onx, smodel, e))

    input = load["data"]
    DF = options.pop('DF', False)
    if DF:
        inputs = {c: input[c].values for c in input.columns}
        for k in inputs:
            if inputs[k].dtype == numpy.float64:
                inputs[k] = inputs[k].astype(numpy.float32)
            inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))
    else:
        if isinstance(input, dict):
            inputs = input
        elif isinstance(input, (list, numpy.ndarray, pandas.DataFrame)):
            inp = sess.get_inputs()
            if len(inp) == len(input):
                inputs = {i.name: v for i, v in zip(inp, input)}
            elif len(inp) == 1:
                inputs = {inp[0].name: input}
            elif isinstance(input, numpy.ndarray):
                shape = sum(i.shape[1] if len(i.shape) == 2 else i.shape[0]
                            for i in inp)
                if shape == input.shape[1]:
                    inputs = {n.name: input[:, i] for i, n in enumerate(inp)}
                else:
                    raise OnnxBackendAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original shape {1}, onnx='{2}'"
                        .format(len(inp), input.shape, onx))
            elif isinstance(input, list):
                try:
                    array_input = numpy.array(input)
                except Exception:  # pragma no cover
                    raise OnnxBackendAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original {1}, onnx='{2}'"
                        .format(len(inp), len(input), onx))
                shape = sum(i.shape[1] for i in inp)
                if shape == array_input.shape[1]:
                    inputs = {}
                    c = 0
                    for i, n in enumerate(inp):
                        d = c + n.shape[1]
                        inputs[n.name] = _create_column(
                            [row[c:d] for row in input], n.type)
                        c = d
                else:
                    raise OnnxBackendAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original shape {1}, onnx='{2}'*"
                        .format(len(inp), array_input.shape, onx))
            elif isinstance(input, pandas.DataFrame):
                try:
                    array_input = numpy.array(input)
                except Exception:  # pragma no cover
                    raise OnnxBackendAssertionError(
                        "Wrong number of inputs onnx {0} != "
                        "original {1}, onnx='{2}'"
                        .format(len(inp), len(input), onx))
                shape = sum(i.shape[1] for i in inp)
                if shape == array_input.shape[1]:
                    inputs = {}
                    c = 0
                    for i, n in enumerate(inp):
                        d = c + n.shape[1]
                        inputs[n.name] = _create_column(
                            input.iloc[:, c:d], n.type)
                        c = d
                else:
                    raise OnnxBackendAssertionError(
                        "Wrong number of inputs onnx {0}={1} columns != "
                        "original shape {2}, onnx='{3}'*"
                        .format(len(inp), shape, array_input.shape, onx))
            else:
                raise OnnxBackendAssertionError(
                    "Wrong type of inputs onnx {0}, onnx='{1}'".format(
                        type(input), onx))
        else:
            raise OnnxBackendAssertionError(
                "Dict or list is expected, not {0}".format(type(input)))

        for k in inputs:
            if isinstance(inputs[k], list):
                inputs[k] = numpy.array(inputs[k])

    options.pop('SklCol', False)  # unused here but in dump_data_and_model

    if verbose:  # pragma no cover
        print("[compare_runtime] type(inputs)={} len={} names={}".format(
            type(input), len(inputs), list(sorted(inputs))))
    if verbose:  # pragma no cover
        run_options = {'verbose': 2, 'fLOG': print}
    else:
        run_options = {}
    try:
        output = sess.run(None, inputs, **run_options)
        lambda_onnx = lambda: sess.run(None, inputs)  # noqa
        if verbose:  # pragma no cover
            import pprint
            pprint.pprint(output)
    except ExpectedAssertionError as expe:  # pragma no cover
        raise expe
    except (RuntimeError, OrtInvalidArgument) as e:  # pragma no cover
        if intermediate_steps:
            sess.run(None, inputs, verbose=3, fLOG=print)
        if "-Fail" in onx:
            raise ExpectedAssertionError(
                "onnxruntime cannot compute the prediction for '{0}'".
                format(onx))
        else:
            if verbose:  # pragma no cover
                model = onnx.load(onx)
                smodel = "\nJSON ONNX\n" + str(model)
            else:
                smodel = ""
            import pprint
            raise OnnxBackendAssertionError(
                "onnxruntime cannot compute the prediction"
                " for '{0}' due to {1}{2}\n{3}"
                .format(onx, e, smodel, pprint.pformat(inputs)))
    except Exception as e:  # pragma no cover
        raise OnnxBackendAssertionError(
            "Unable to run onnx '{0}' due to {1}".format(onx, e))
    if verbose:  # pragma no cover
        print("[compare_runtime] done type={}".format(type(output)))

    output0 = output.copy()

    if comparable_outputs:
        cmp_exp = [load["expected"][o] for o in comparable_outputs]
        cmp_out = [output[o] for o in comparable_outputs]
    else:
        cmp_exp = load["expected"]
        cmp_out = output

    try:
        _compare_expected(cmp_exp, cmp_out, sess, onx,
                          decimal=decimal, verbose=verbose,
                          classes=classes, **options)
    except ExpectedAssertionError as expe:  # pragma no cover
        raise expe
    except Exception as e:  # pragma no cover
        if verbose:  # pragma no cover
            model = onnx.load(onx)
            smodel = "\nJSON ONNX\n" + str(model)
        else:
            smodel = ""
        raise OnnxBackendAssertionError(
            "Model '{0}' has discrepencies.\n{1}: {2}{3}".format(
                onx, type(e), e, smodel))

    return output0, lambda_onnx
