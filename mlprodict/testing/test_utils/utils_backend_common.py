"""
@file
@brief Inspired from :epkg:`sklearn-onnx`, handles two backends.
"""
import os
import pickle
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.sparse.csr import csr_matrix
import pandas
from ...onnxrt.ops_cpu.op_zipmap import ArrayZipMapDictionary


class ExpectedAssertionError(Exception):
    """
    Expected failure.
    """
    pass


class OnnxBackendAssertionError(AssertionError):
    """
    Expected failure.
    """
    pass


class OnnxBackendMissingNewOnnxOperatorException(OnnxBackendAssertionError):
    """
    Raised when :epkg:`onnxruntime` or :epkg:`mlprodict`
    does not implement a new operator
    defined in the latest onnx.
    """
    pass


class OnnxRuntimeMissingNewOnnxOperatorException(OnnxBackendAssertionError):
    """
    Raised when a new operator was added but cannot be found.
    """
    pass


def evaluate_condition(backend, condition):
    """
    Evaluates a condition such as
    ``StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.3')``
    """
    if backend == "onnxruntime":  # pragma: no cover
        import onnxruntime  # pylint: disable=W0611
        return eval(condition)  # pylint: disable=W0123
    raise NotImplementedError(  # pragma no cover
        "Not implemented for backend '{0}' and "
        "condition '{1}'.".format(backend, condition))


def is_backend_enabled(backend):
    """
    Tells if a backend is enabled.
    Raises an exception if backend != 'onnxruntime'.
    Unit tests only test models against this backend.
    """
    if backend == "onnxruntime":
        try:
            import onnxruntime  # pylint: disable=W0611
            return True
        except ImportError:  # pragma no cover
            return False
    if backend == "python":
        return True
    raise NotImplementedError(  # pragma no cover
        "Not implemented for backend '{0}'".format(backend))


def load_data_and_model(items_as_dict, **context):
    """
    Loads every file in a dictionary {key: filename}.
    The extension is either *pkl* and *onnx* and determines
    how it it loaded. If the value is not a string,
    the function assumes it was already loaded.
    """
    res = {}
    for k, v in items_as_dict.items():
        if isinstance(v, str):
            if os.path.splitext(v)[-1] == ".pkl":
                with open(v, "rb") as f:  # pragma: no cover
                    try:
                        bin = pickle.load(f)
                    except ImportError as e:
                        if '.model.' in v:
                            continue
                        raise ImportError(  # pylint: disable=W0707
                            "Unable to load '{0}' due to {1}".format(v, e))
                    res[k] = bin
            else:
                res[k] = v
        else:
            res[k] = v
    return res


def extract_options(name):
    """
    Extracts comparison option from filename.
    As example, ``Binarizer-SkipDim1`` means
    options *SkipDim1* is enabled.
    ``(1, 2)`` and ``(2,)`` are considered equal.
    Available options: see :func:`dump_data_and_model`.
    """
    opts = name.replace("\\", "/").split("/")[-1].split('.')[0].split('-')
    if len(opts) == 1:
        return {}
    res = {}
    for opt in opts[1:]:
        if opt in ("SkipDim1", "OneOff", "NoProb", "NoProbOpp",
                   "Dec4", "Dec3", "Dec2", 'Svm',
                   'Out0', 'Reshape', 'SklCol', 'DF', 'OneOffArray'):
            res[opt] = True
        else:
            raise NameError("Unable to parse option '{}'".format(
                opts[1:]))  # pragma no cover
    return res


def compare_outputs(expected, output, verbose=False, **kwargs):
    """
    Compares expected values and output.
    Returns None if no error, an exception message otherwise.
    """
    SkipDim1 = kwargs.pop("SkipDim1", False)
    NoProb = kwargs.pop("NoProb", False)
    NoProbOpp = kwargs.pop("NoProbOpp", False)
    Dec4 = kwargs.pop("Dec4", False)
    Dec3 = kwargs.pop("Dec3", False)
    Dec2 = kwargs.pop("Dec2", False)
    Disc = kwargs.pop("Disc", False)
    Mism = kwargs.pop("Mism", False)

    if Dec4:
        kwargs["decimal"] = min(kwargs["decimal"], 4)
    if Dec3:
        kwargs["decimal"] = min(kwargs["decimal"], 3)
    if Dec2:
        kwargs["decimal"] = min(kwargs["decimal"], 2)  # pragma: no cover
    if isinstance(expected, numpy.ndarray) and isinstance(
            output, numpy.ndarray):
        if SkipDim1:
            # Arrays like (2, 1, 2, 3) becomes (2, 2, 3)
            # as one dimension is useless.
            expected = expected.reshape(
                tuple([d for d in expected.shape if d > 1]))
            output = output.reshape(
                tuple([d for d in expected.shape if d > 1]))
        if NoProb or NoProbOpp:
            # One vector is (N,) with scores, negative for class 0
            # positive for class 1
            # The other vector is (N, 2) score in two columns.
            if len(output.shape) == 2 and output.shape[1] == 2 and len(
                    expected.shape) == 1:
                output = output[:, 1]
                if NoProbOpp:
                    output = -output
            elif len(output.shape) == 1 and len(expected.shape) == 1:
                pass
            elif len(expected.shape) == 1 and len(output.shape) == 2 and \
                    expected.shape[0] == output.shape[0] and \
                    output.shape[1] == 1:
                output = output[:, 0]
                if NoProbOpp:
                    output = -output
            elif expected.shape != output.shape:
                raise NotImplementedError("Shape mismatch: {0} != {1}".format(  # pragma no cover
                    expected.shape, output.shape))
        if len(expected.shape) == 1 and len(
                output.shape) == 2 and output.shape[1] == 1:
            output = output.ravel()
        if len(output.shape) == 3 and output.shape[0] == 1 and len(
                expected.shape) == 2:
            output = output.reshape(output.shape[1:])
        if expected.dtype in (numpy.str_, numpy.dtype("<U1"),
                              numpy.dtype("<U3")):
            try:
                assert_array_equal(expected, output, verbose=verbose)
            except Exception as e:  # pylint: disable=W0703
                if Disc:  # pragma no cover
                    # Bug to be fixed later.
                    return ExpectedAssertionError(str(e))
                else:  # pragma no cover
                    return OnnxBackendAssertionError(str(e))
        else:
            try:
                assert_array_almost_equal(expected,
                                          output,
                                          verbose=verbose,
                                          **kwargs)
            except (RuntimeError, AssertionError) as e:  # pragma no cover
                longer = "\n--EXPECTED--\n{0}\n--OUTPUT--\n{1}".format(
                    expected, output) if verbose else ""
                expected_ = numpy.asarray(expected).ravel()
                output_ = numpy.asarray(output).ravel()
                if len(expected_) == len(output_):
                    if numpy.issubdtype(expected_.dtype, numpy.floating):
                        diff = numpy.abs(expected_ - output_).max()
                    else:
                        diff = max((1 if ci != cj else 0)
                                   for ci, cj in zip(expected_, output_))
                    if diff == 0:
                        return None
                elif Mism:
                    return ExpectedAssertionError(
                        "dimension mismatch={0}, {1}\n{2}{3}".format(
                            expected.shape, output.shape, e, longer))
                else:
                    return OnnxBackendAssertionError(
                        "dimension mismatch={0}, {1}\n{2}{3}".format(
                            expected.shape, output.shape, e, longer))
                if Disc:
                    # Bug to be fixed later.
                    return ExpectedAssertionError(
                        "max-diff={0}\n--expected--output--\n{1}{2}".format(
                            diff, e, longer))
                return OnnxBackendAssertionError(
                    "max-diff={0}\n--expected--output--\n{1}{2}".format(
                        diff, e, longer))
    else:
        return OnnxBackendAssertionError(  # pragma: no cover
            "Unexpected types {0} != {1}".format(
                type(expected), type(output)))
    return None


def _post_process_output(res):
    """
    Applies post processings before running the comparison
    such as changing type from list to arrays.
    """
    if isinstance(res, list):
        if len(res) == 0:
            return res
        if len(res) == 1:
            return _post_process_output(res[0])
        if isinstance(res[0], numpy.ndarray):
            return numpy.array(res)
        if isinstance(res[0], dict):
            return pandas.DataFrame(res).values
        ls = [len(r) for r in res]
        mi = min(ls)
        if mi != max(ls):
            raise NotImplementedError(  # pragma no cover
                "Unable to postprocess various number of "
                "outputs in [{0}, {1}]"
                .format(min(ls), max(ls)))
        if mi > 1:
            output = []
            for i in range(mi):
                output.append(_post_process_output([r[i] for r in res]))
            return output
        if isinstance(res[0], list):
            # list of lists
            if isinstance(res[0][0], list):
                return numpy.array(res)
            if len(res[0]) == 1 and isinstance(res[0][0], dict):
                return _post_process_output([r[0] for r in res])
            if len(res) == 1:
                return res
            if len(res[0]) != 1:
                raise NotImplementedError(  # pragma no cover
                    "Not conversion implemented for {0}".format(res))
            st = [r[0] for r in res]
            return numpy.vstack(st)
        return res
    return res


def _create_column(values, dtype):
    "Creates a column from values with dtype"
    if str(dtype) == "tensor(int64)":
        return numpy.array(values, dtype=numpy.int64)
    if str(dtype) == "tensor(float)":
        return numpy.array(values, dtype=numpy.float32)
    if str(dtype) in ("tensor(double)", "tensor(float64)"):
        return numpy.array(values, dtype=numpy.float64)
    if str(dtype) in ("tensor(string)", "tensor(str)"):
        return numpy.array(values, dtype=numpy.str_)
    raise OnnxBackendAssertionError(
        "Unable to create one column from dtype '{0}'".format(dtype))


def _compare_expected(expected, output, sess, onnx_model,
                      decimal=5, verbose=False, classes=None,
                      **kwargs):
    """
    Compares the expected output against the runtime outputs.
    This is specific to :epkg:`onnxruntime` or :epkg:`mlprodict`.
    """
    tested = 0
    if isinstance(expected, list):
        if isinstance(output, list):
            if 'Out0' in kwargs:
                expected = expected[:1]
                output = output[:1]
                del kwargs['Out0']
            if 'Reshape' in kwargs:
                del kwargs['Reshape']
                output = numpy.hstack(output).ravel()
                output = output.reshape(
                    (len(expected), len(output.ravel()) // len(expected)))
            if len(expected) != len(output):
                raise OnnxBackendAssertionError(  # pragma no cover
                    "Unexpected number of outputs '{0}', expected={1}, got={2}"
                    .format(onnx_model, len(expected), len(output)))
            for exp, out in zip(expected, output):
                _compare_expected(exp, out, sess, onnx_model, decimal=5, verbose=verbose,
                                  classes=classes, **kwargs)
                tested += 1
        else:
            raise OnnxBackendAssertionError(  # pragma no cover
                "Type mismatch for '{0}', output type is {1}".format(
                    onnx_model, type(output)))
    elif isinstance(expected, dict):
        if not isinstance(output, dict):
            raise OnnxBackendAssertionError(  # pragma no cover
                "Type mismatch for '{0}'".format(onnx_model))
        for k, v in output.items():
            if k not in expected:
                continue
            msg = compare_outputs(
                expected[k], v, decimal=decimal, verbose=verbose, **kwargs)
            if msg:
                raise OnnxBackendAssertionError(  # pragma no cover
                    "Unexpected output '{0}' in model '{1}'\n{2}".format(
                        k, onnx_model, msg))
            tested += 1
    elif isinstance(expected, numpy.ndarray):
        if isinstance(output, list):
            if expected.shape[0] == len(output) and isinstance(
                    output[0], dict):
                if isinstance(output, ArrayZipMapDictionary):
                    output = pandas.DataFrame(list(output))
                else:
                    output = pandas.DataFrame(output)
                output = output[list(sorted(output.columns))]
                output = output.values
        if isinstance(output, (dict, list)):
            if len(output) != 1:  # pragma: no cover
                ex = str(output)
                if len(ex) > 170:
                    ex = ex[:170] + "..."
                raise OnnxBackendAssertionError(
                    "More than one output when 1 is expected "
                    "for onnx '{0}'\n{1}"
                    .format(onnx_model, ex))
            output = output[-1]
        if not isinstance(output, numpy.ndarray):
            raise OnnxBackendAssertionError(  # pragma no cover
                "output must be an array for onnx '{0}' not {1}".format(
                    onnx_model, type(output)))
        if (classes is not None and (
                expected.dtype == numpy.str_ or expected.dtype.char == 'U')):
            try:
                output = numpy.array([classes[cl] for cl in output])
            except IndexError as e:  # pragma no cover
                raise RuntimeError('Unable to handle\n{}\n{}\n{}'.format(
                    expected, output, classes)) from e
        msg = compare_outputs(
            expected, output, decimal=decimal, verbose=verbose, **kwargs)
        if isinstance(msg, ExpectedAssertionError):
            raise msg  # pylint: disable=E0702
        if msg:
            raise OnnxBackendAssertionError(  # pragma no cover
                "Unexpected output in model '{0}'\n{1}".format(onnx_model, msg))
        tested += 1
    else:
        if isinstance(expected, csr_matrix):
            # DictVectorizer
            one_array = numpy.array(output)
            dense = numpy.asarray(expected.todense())
            msg = compare_outputs(dense, one_array, decimal=decimal,
                                  verbose=verbose, **kwargs)
            if msg:
                raise OnnxBackendAssertionError(  # pragma no cover
                    "Unexpected output in model '{0}'\n{1}".format(onnx_model, msg))
            tested += 1
        else:
            raise OnnxBackendAssertionError(  # pragma no cover
                "Unexpected type for expected output ({1}) and onnx '{0}'".
                format(onnx_model, type(expected)))
    if tested == 0:
        raise OnnxBackendAssertionError(  # pragma no cover
            "No test for onnx '{0}'".format(onnx_model))
