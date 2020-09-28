"""
@file
@brief Utilies to test script from :epkg:`scikit-learn` documentation.
"""
import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import pprint
import numpy
from sklearn.base import BaseEstimator
from .verify_code import verify_code


class MissingVariableError(RuntimeError):
    """
    Raised when a variable is missing.
    """
    pass


def _clean_script(content):
    """
    Comments out all lines containing ``.show()``.
    """
    new_lines = []
    for line in content.split('\n'):
        if '.show()' in line or 'sys.exit' in line:
            new_lines.append("# " + line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def _enumerate_fit_info(fits):
    """
    Extracts the name of the fitted models and the data
    used to train it.
    """
    for fit in fits:
        chs = fit['children']
        if len(chs) < 2:
            # unable to extract the needed information
            continue  # pragma: no cover
        model = chs[0]['str']
        if model.endswith('.fit'):
            model = model[:-4]
        args = [ch['str'] for ch in chs[1:]]
        yield model, args


def _try_onnx(loc, model_name, args_name, **options):
    """
    Tries onnx conversion.

    @param  loc         available variables
    @param  model_name  model name among these variables
    @param  args_name   arguments name among these variables
    @param  options     additional options for the conversion
    @return             onnx model
    """
    from ..onnx_conv import to_onnx
    if model_name not in loc:
        raise MissingVariableError("Unable to find model '{}' in {}".format(
            model_name, ", ".join(sorted(loc))))
    if args_name[0] not in loc:
        raise MissingVariableError("Unable to find data '{}' in {}".format(
            args_name[0], ", ".join(sorted(loc))))
    model = loc[model_name]
    X = loc[args_name[0]]
    dtype = options.get('dtype', numpy.float32)
    Xt = X.astype(dtype)
    onx = to_onnx(model, Xt, **options)
    args = dict(onx=onx, model=model, X=Xt)
    return onx, args


def verify_script(file_or_name, try_onnx=True, existing_loc=None,
                  **options):
    """
    Checks that models fitted in an example from :epkg:`scikit-learn`
    documentation can be converted into :epkg:`ONNX`.

    @param      file_or_name        file or string
    @param      try_onnx            try the onnx conversion
    @param      existing_loc        existing local variables
    @param      options             conversion options
    @return                         list of converted models
    """
    if '\n' not in file_or_name and os.path.exists(file_or_name):
        filename = file_or_name
        with open(file_or_name, 'r', encoding='utf-8') as f:
            content = f.read()
    else:  # pragma: no cover
        content = file_or_name
        filename = "<string>"

    # comments out .show()
    content = _clean_script(content)

    # look for fit or predict expressions
    _, node = verify_code(content, exc=False)
    fits = node._fits
    models_args = list(_enumerate_fit_info(fits))

    # execution
    obj = compile(content, filename, 'exec')
    glo = globals().copy()
    loc = {}
    if existing_loc is not None:
        loc.update(existing_loc)
        glo.update(existing_loc)
    out = StringIO()
    err = StringIO()

    with redirect_stdout(out):
        with redirect_stderr(err):
            exec(obj, glo, loc)  # pylint: disable=W0122

    # filter out values
    cls = (BaseEstimator, numpy.ndarray)
    loc_fil = {k: v for k, v in loc.items() if isinstance(v, cls)}
    glo_fil = {k: v for k, v in glo.items() if k not in {'__builtins__'}}
    onx_info = []

    # onnx
    if try_onnx:
        if len(models_args) == 0:
            raise MissingVariableError(
                "No detected trained model in '{}'\n{}\n--LOCALS--\n{}".format(
                    filename, content, pprint.pformat(loc_fil)))
        for model_args in models_args:
            try:
                onx, args = _try_onnx(loc_fil, *model_args, **options)
            except MissingVariableError as e:
                raise MissingVariableError("Unable to find variable in '{}'\n{}".format(
                    filename, pprint.pformat(fits))) from e
            loc_fil[model_args[0] + "_onnx"] = onx
            onx_info.append(args)

    # final results
    return dict(locals=loc_fil, globals=glo_fil,
                stdout=out.getvalue(),
                stderr=err.getvalue(),
                onx_info=onx_info)
