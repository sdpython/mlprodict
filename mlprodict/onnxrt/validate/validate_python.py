"""
@file
@brief Helpers to validate python code.
"""
import pickle
import numpy
from scipy.spatial.distance import cdist  # pylint: disable=E0611
from scipy.special import expit, erf  # pylint: disable=E0611
from scipy.linalg import solve  # pylint: disable=E0611
from ...tools.code_helper import make_callable


def _make_callable(fct, obj, code, gl, debug):
    """
    Same function as @see fn make_callable but deals with
    function which an undefined number of arguments.
    """
    def pyrt_Concat_(*inputs, axis=0):
        return numpy.concatenate(inputs, axis=axis)

    if fct == "pyrt_Concat":
        return pyrt_Concat_
    return make_callable(fct, obj, code, gl, debug)


def validate_python_inference(oinf, inputs, tolerance=0.):
    """
    Validates the code produced by method :meth:`to_python
    <mlprodict.onnxrt.onnx_inference_exports.OnnxInferenceExport.to_python>`.
    The function compiles and executes the code
    given as an argument and compares the results to
    what *oinf* returns. This function is mostly used for
    unit testing purpose but it is not robust enough
    to handle all cases.

    @param      oinf        @see cl OnnxInference
    @param      inputs      inputs as dictionary
    @param      tolerance   discrepencies must be below or equal to
                            this theshold

    The function fails if the expected output are not the same.
    """
    from ..ops_cpu.op_argmax import _argmax
    from ..ops_cpu.op_argmin import _argmin
    from ..ops_cpu.op_celu import _vcelu1

    cd = oinf.to_python()
    code = cd['onnx_pyrt_main.py']

    exp = oinf.run(inputs)
    if not isinstance(exp, dict):
        raise TypeError("exp is not a dictionary by '{}'.".format(type(exp)))
    if len(exp) == 0:
        raise ValueError("No result to compare.")
    inps = ['{0}={0}'.format(k) for k in sorted(inputs)]
    code += "\n".join(['', '', 'opi = OnnxPythonInference()',
                       'res = opi.run(%s)' % ', '.join(inps)])

    cp = compile(code, "<string>", mode='exec')
    pyrt_fcts = [_ for _ in cp.co_names if _.startswith("pyrt_")]
    fcts_local = {}

    gl = {'numpy': numpy, 'pickle': pickle,
          'expit': expit, 'erf': erf, 'cdist': cdist,
          '_argmax': _argmax, '_argmin': _argmin,
          '_vcelu1': _vcelu1, 'solve': solve}

    for fct in pyrt_fcts:
        for obj in cp.co_consts:
            if isinstance(obj, str):
                continue
            sobj = str(obj)
            if '<string>' in sobj and fct in sobj:
                fcts_local[fct] = _make_callable(fct, obj, code, gl, False)

    gl.update(fcts_local)
    loc = inputs
    try:
        exec(cp, gl, loc)  # pylint: disable=W0122
    except (NameError, TypeError, SyntaxError) as e:
        raise RuntimeError(
            "Unable to execute code\n-----\n{}".format(code)) from e

    got = loc['res']
    keys = list(sorted(exp))
    if isinstance(got, numpy.ndarray) and len(keys) == 1:
        got = {keys[0]: got}

    if not isinstance(got, dict):
        raise TypeError("got is not a dictionary by '{}'.".format(type(got)))
    if len(got) != len(exp):
        raise RuntimeError(
            "Different number of results.\nexp: {}\ngot: {}".format(
                ", ".join(sorted(exp)), ", ".join(sorted(got))))

    if keys != list(sorted(got)):
        raise RuntimeError(
            "Different result names.\nexp: {}\ngot: {}".format(
                ", ".join(sorted(exp)), ", ".join(sorted(got))))

    for k in keys:
        e = exp[k]
        g = got[k]
        if isinstance(e, numpy.ndarray):
            if e.shape != g.shape:
                raise ValueError(
                    "Shapes are different {} != {}.".format(e.shape, g.shape))
            diff = 0
            for a, b in zip(e.ravel(), g.ravel()):
                if a == b:
                    continue
                if (isinstance(a, float) and isinstance(b, float) and
                        numpy.isnan(a) and numpy.isnan(b)):
                    continue
                diff = max(diff, abs(a - b))
            if diff > tolerance:
                raise ValueError(
                    "Values are different (max diff={}>{})\n--EXP--\n{}\n--GOT--"
                    "\n{}\n--\n{}".format(diff, tolerance, e, g, code))
        else:
            raise NotImplementedError(
                "Unable to compare values of type '{}'.".format(type(e)))
