"""
@file
@brief Helpers to validate python code.
"""
import pickle
import types
import re
import numpy
from scipy.special import expit  # pylint: disable=E0611


def _make_callable(fct, obj, code, gl):
    """
    Creates a callable function able to
    cope with default values as the combination
    of functions *compile* and *exec* does not seem
    able to take them into account.

    @param      fct     function name
    @param      obj     output of function *compile*
    @param      code    code including the signature
    @param      gl      global context
    @return             callable functions
    """
    cst = "def " + fct + "("
    sig = None
    for line in code.split('\n'):
        if line.startswith(cst):
            sig = line
            break
    if sig is None:
        raise ValueError(
            "Unable to find function '{}' in\n{}".format(fct, code))
    reg = re.compile("([a-z][A-Za-z_0-9]*)=([0-9.e+-]+)")
    fall = reg.findall(sig)
    defs = []
    for name, value in fall:
        f = float(value)
        if int(f) == f:
            f = int(f)
        defs.append((name, f))
    res = types.FunctionType(obj, gl, fct, tuple(_[1] for _ in defs))
    return res


def validate_python_inference(oinf, inputs):
    """
    Validates the code produced by method :meth:`to_python
    <mlprodict.onnxrt.onnx_inference_exports.OnnxInferenceExport.to_python>`.
    The function compiles and executes the code
    given as an argument and compares the results to
    what *oinf* returns.

    @param      oinf        @see cl OnnxInference
    @param      inputs      inputs as dictionary

    The function fails if the expected output are not the same.
    """
    from ..ops_cpu.op_argmax import _argmax
    from ..ops_cpu.op_argmin import _argmin

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
          'expit': expit, '_argmax': _argmax,
          '_argmin': _argmin}

    for fct in pyrt_fcts:
        for obj in cp.co_consts:
            if isinstance(obj, str):
                continue
            sobj = str(obj)
            if '<string>' in sobj and fct in sobj:
                fcts_local[fct] = _make_callable(fct, obj, code, gl)

    gl.update(fcts_local)
    loc = inputs
    try:
        exec(cp, gl, loc)  # pylint: disable=W0122
    except (NameError, TypeError, SyntaxError) as e:
        raise RuntimeError(
            "Unable to compile code\n-----\n{}".format(code)) from e

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
            if diff > 0:
                raise ValueError(
                    "Values are different (max diff={})\n--EXP--\n{}\n--GOT--"
                    "\n{}\n--\n{}".format(diff, e, g, code))
        else:
            raise NotImplementedError(
                "Unable to compare values of type '{}'.".format(type(e)))
