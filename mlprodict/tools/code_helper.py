"""
@file
@brief A couple of tools unrelated to what the package does.
"""
import keyword
import re
import types
import numpy


def change_style(name):
    """
    Switches from *AaBb* into *aa_bb*.

    @param      name    name to convert
    @return             converted name
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return s2 if not keyword.iskeyword(s2) else s2 + "_"


def numpy_min_max(x, fct):
    """
    Returns the minimum of an array.
    Deals with text as well.
    """
    try:
        if hasattr(x, 'todense'):
            x = x.todense()
        if x.dtype.kind.lower() not in 'uc':
            return fct(x)
        try:  # pragma: no cover
            x = x.ravel()
        except AttributeError:  # pragma: no cover
            pass
        keep = list(filter(lambda s: isinstance(s, str), x))
        if len(keep) == 0:  # pragma: no cover
            return numpy.nan
        keep.sort()
        val = keep[0]
        if len(val) > 10:  # pragma: no cover
            val = val[:10] + '...'
        return "%r" % val
    except (ValueError, TypeError):
        return '?'


def numpy_min(x):
    """
    Returns the maximum of an array.
    Deals with text as well.
    """
    return numpy_min_max(x, lambda x: x.min())


def numpy_max(x):
    """
    Returns the maximum of an array.
    Deals with text as well.
    """
    return numpy_min_max(x, lambda x: x.max())


def debug_print(k, obj, printed):
    """
    Displays informations on an object.

    @param      k       name
    @param      obj     object
    @param      printed memorizes already printed object
    """
    if k not in printed:
        printed[k] = obj
        if hasattr(obj, 'shape'):
            print("-='{}' shape={} dtype={} min={} max={}{}".format(
                  k, obj.shape, obj.dtype, numpy_min(obj),
                  numpy_max(obj),
                  ' (sparse)' if 'coo_matrix' in str(type(obj)) else ''))
        elif (isinstance(obj, list) and len(obj) > 0 and
                not isinstance(obj[0], dict)):
            print("-='{}' list len={} min={} max={}".format(
                  k, len(obj), min(obj), max(obj)))
        else:
            print("-='{}' type={}".format(k, type(obj)))


def make_callable(fct, obj, code, gl, debug):
    """
    Creates a callable function able to
    cope with default values as the combination
    of functions *compile* and *exec* does not seem
    able to take them into account.

    @param      fct     function name
    @param      obj     output of function *compile*
    @param      code    code including the signature
    @param      gl      context (local and global)
    @param      debug   add debug function
    @return             callable functions
    """
    cst = "def " + fct + "("
    sig = None
    for line in code.split('\n'):
        if line.startswith(cst):
            sig = line
            break
    if sig is None:  # pragma: no cover
        raise ValueError(
            "Unable to find function '{}' in\n{}".format(fct, code))
    reg = re.compile(
        "([a-z][A-Za-z_0-9]*)=((None)|(False)|(True)|([0-9.e+-]+))")
    fall = reg.findall(sig)
    defs = []
    for name_value in fall:
        name = name_value[0]
        value = name_value[1]
        if value == 'None':
            defs.append((name, None))
            continue
        if value == 'True':
            defs.append((name, True))
            continue
        if value == 'False':
            defs.append((name, False))
            continue
        f = float(value)
        if int(f) == f:
            f = int(f)
        defs.append((name, f))

    # debug
    if debug:
        gl = gl.copy()
        gl['debug_print'] = debug_print
        gl['print'] = print
    # specific
    if "value=array([0.], dtype=float32)" in sig:
        defs.append(('value', numpy.array([0.], dtype=numpy.float32)))
    res = types.FunctionType(obj, gl, fct, tuple(_[1] for _ in defs))
    if res.__defaults__ != tuple(_[1] for _ in defs):  # pylint: disable=E1101
        # See https://docs.python.org/3/library/inspect.html
        # See https://stackoverflow.com/questions/11291242/python-dynamically-create-function-at-runtime
        lines = [str(sig)]
        for name in ['co_argcount', 'co_cellvars', 'co_code', 'co_consts', 'co_filename',
                     'co_firstlineno', 'co_flags', 'co_freevars', 'co_kwonlyargcount',
                     'co_lnotab', 'co_name', 'co_names', 'co_nlocals', 'co_stacksize',
                     'co_varnames']:
            v = getattr(res.__code__, name, None)  # pylint: disable=E1101
            if v is not None:
                lines.append('%s=%r' % (name, v))
        raise RuntimeError(  # pragma: no cover
            "Defaults values of function '{}' (defaults={}) are missing.\nDefault: "
            "{}\n{}\n----\n{}".format(
                fct, res.__defaults__, defs, "\n".join(lines), code))  # pylint: disable=E1101
    return res
