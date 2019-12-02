"""
@file
@brief Helpers to validate python code.
"""
import pickle
import types
import numpy


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
    fcts = {}
    for fct in pyrt_fcts:
        for obj in cp.co_consts:
            if isinstance(obj, str):
                continue
            sobj = str(obj)
            if '<string>' in sobj and fct in sobj:
                fcts[fct] = obj
                break

    fcts_local = {}
    gl = {'numpy': numpy, 'pickle': pickle}
    for k, v in fcts.items():
        fcts_local[k] = types.FunctionType(v, gl, k)

    gl.update(fcts_local)
    loc = inputs
    exec(cp, gl, loc)  # pylint: disable=W0122

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
            if e.ravel().tolist() != g.ravel().tolist():
                raise ValueError("Values are different")
        else:
            raise NotImplementedError(
                "Unable to compare values of type '{}'.".format(type(e)))
