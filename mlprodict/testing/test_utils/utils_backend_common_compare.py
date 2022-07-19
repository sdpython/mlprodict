"""
@file
@brief Inspired from sklearn-onnx, handles two backends.
"""
import numpy
import onnx
import pandas
from .utils_backend_common import (
    load_data_and_model, extract_options,
    ExpectedAssertionError, OnnxBackendAssertionError,
    OnnxRuntimeMissingNewOnnxOperatorException,
    _compare_expected, _create_column)


def compare_runtime_session(  # pylint: disable=R0912
        cls_session, test, decimal=5, options=None,
        verbose=False, context=None, comparable_outputs=None,
        intermediate_steps=False, classes=None,
        disable_optimisation=False):
    """
    The function compares the expected output (computed with
    the model before being converted to ONNX) and the ONNX output
    produced with module :epkg:`onnxruntime` or :epkg:`mlprodict`.

    :param cls_session: inference session instance (like @see cl OnnxInference)
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
    lambda_onnx = None
    if context is None:
        context = {}
    load = load_data_and_model(test, **context)
    if verbose:  # pragma no cover
        print(f"[compare_runtime] test '{test['onnx']}' loaded")

    onx = test['onnx']

    if options is None:
        if isinstance(onx, str):
            options = extract_options(onx)
        else:
            options = {}
    elif options is None:
        options = {}
    elif not isinstance(options, dict):
        raise TypeError(  # pragma no cover
            "options must be a dictionary.")

    if verbose:  # pragma no cover
        print(f"[compare_runtime] InferenceSession('{onx}')")

    runtime_options = dict(disable_optimisation=disable_optimisation)
    try:
        sess = cls_session(onx, runtime_options=runtime_options)
    except TypeError as et:  # pragma: no cover
        raise TypeError(  # pylint: disable=W0707
            f"Wrong signature for '{cls_session.__name__}' ({et}).")
    except ExpectedAssertionError as expe:  # pragma no cover
        raise expe
    except Exception as e:  # pylint: disable=W0703
        if "CannotLoad" in options:  # pragma no cover
            raise ExpectedAssertionError(  # pylint: disable=W0707
                f"Unable to load onnx '{onx}' due to\n{e}")
        else:  # pragma no cover
            if verbose:  # pragma no cover
                model = onnx.load(onx)
                smodel = "\nJSON ONNX\n" + str(model)
            else:
                smodel = ""
            if ("NOT_IMPLEMENTED : Could not find an implementation "
                    "for the node" in str(e)):
                # onnxruntime does not implement a specific node yet.
                raise OnnxRuntimeMissingNewOnnxOperatorException(  # pylint: disable=W0707
                    "{3} does not implement a new operator "
                    "'{0}'\n{1}\nONNX\n{2}".format(
                        onx, e, smodel, cls_session))
            if "NOT_IMPLEMENTED : Failed to find kernel" in str(e):
                # onnxruntime does not implement a specific node yet
                # in the kernel included in onnxruntime.
                raise OnnxBackendAssertionError(  # pylint: disable=W0707
                    "{3} misses a kernel for operator "
                    "'{0}'\n{1}\nONNX\n{2}".format(
                        onx, e, smodel, cls_session))
            raise OnnxBackendAssertionError(  # pylint: disable=W0707
                f"Unable to load onnx '{onx}'\nONNX\n{smodel}\n{e}")

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
            outs = sess.get_outputs()
            if len(outs) == 0:
                raise OnnxBackendAssertionError(  # pragma: no cover
                    "Wrong number of outputs, onnx='{2}'".format(onx))
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
                    raise OnnxBackendAssertionError(  # pragma: no cover
                        "Wrong number of inputs onnx {0} != "
                        "original shape {1}, onnx='{2}'"
                        .format(len(inp), input.shape, onx))
            elif isinstance(input, list):
                try:
                    array_input = numpy.array(input)
                except Exception:  # pragma no cover
                    raise OnnxBackendAssertionError(  # pylint: disable=W0707
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
                    raise OnnxBackendAssertionError(  # pragma no cover
                        "Wrong number of inputs onnx {0} != "
                        "original shape {1}, onnx='{2}'*"
                        .format(len(inp), array_input.shape, onx))
            elif isinstance(input, pandas.DataFrame):
                try:
                    array_input = numpy.array(input)
                except Exception:  # pragma no cover
                    raise OnnxBackendAssertionError(  # pylint: disable=W0707
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
                    raise OnnxBackendAssertionError(  # pragma no cover
                        "Wrong number of inputs onnx {0}={1} columns != "
                        "original shape {2}, onnx='{3}'*"
                        .format(len(inp), shape, array_input.shape, onx))
            else:
                raise OnnxBackendAssertionError(  # pragma no cover
                    f"Wrong type of inputs onnx {type(input)}, onnx='{onx}'")
        else:
            raise OnnxBackendAssertionError(  # pragma no cover
                f"Dict or list is expected, not {type(input)}")

        for k in inputs:
            if isinstance(inputs[k], list):
                inputs[k] = numpy.array(inputs[k])

    options.pop('SklCol', False)  # unused here but in dump_data_and_model

    if verbose:  # pragma no cover
        print("[compare_runtime] type(inputs)={} len={} names={}".format(
            type(input), len(inputs), list(sorted(inputs))))
    if verbose:  # pragma no cover
        if intermediate_steps:
            run_options = {'verbose': 3, 'fLOG': print}
        else:
            run_options = {'verbose': 2, 'fLOG': print}
    else:
        run_options = {}

    from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
        InvalidArgument as OrtInvalidArgument)

    try:
        try:
            output = sess.run(None, inputs, **run_options)
        except TypeError:  # pragma no cover
            output = sess.run(None, inputs)
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
            raise ExpectedAssertionError(  # pylint: disable=W0707
                f"{cls_session} cannot compute the prediction for '{onx}'")
        else:
            if verbose:  # pragma no cover
                from ...plotting.text_plot import onnx_simple_text_plot
                model = onnx.load(onx)
                smodel = "\nJSON ONNX\n" + onnx_simple_text_plot(
                    model, recursive=True, raise_exc=False)
            else:
                smodel = ""
            import pprint
            raise OnnxBackendAssertionError(  # pylint: disable=W0707
                "{4} cannot compute the predictions"
                " for '{0}' due to {1}{2}\n{3}"
                .format(onx, e, smodel, pprint.pformat(inputs),
                        cls_session))
    except Exception as e:  # pragma no cover
        raise OnnxBackendAssertionError(  # pylint: disable=W0707
            f"Unable to run onnx '{onx}' due to {e}")
    if verbose:  # pragma no cover
        print(f"[compare_runtime] done type={type(output)}")

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
        raise OnnxBackendAssertionError(  # pylint: disable=W0707
            "Model '{}' has discrepencies with cls='{}'.\n{}: {}{}".format(
                onx, sess.__class__.__name__, type(e), e, smodel))

    return output0, lambda_onnx
