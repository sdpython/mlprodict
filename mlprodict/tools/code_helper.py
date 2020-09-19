"""
@file
@brief A couple of tools unrelated to what the package does.
"""
import pickle
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


def numpy_min_max(x, fct, minmax=False):
    """
    Returns the minimum of an array.
    Deals with text as well.
    """
    try:
        if hasattr(x, 'todense'):
            x = x.todense()
        if (x.dtype.kind[0] not in 'Uc' or
                x.dtype in {numpy.uint8}):
            return fct(x)
        try:  # pragma: no cover
            x = x.ravel()
        except AttributeError:  # pragma: no cover
            pass
        keep = list(filter(lambda s: isinstance(s, str), x))
        if len(keep) == 0:  # pragma: no cover
            return numpy.nan
        keep.sort(reverse=minmax)
        val = keep[0]
        if len(val) > 10:  # pragma: no cover
            val = val[:10] + '...'
        return "%r" % val
    except (ValueError, TypeError, AttributeError):
        return '?'


def numpy_min(x):
    """
    Returns the maximum of an array.
    Deals with text as well.
    """
    return numpy_min_max(x, lambda x: x.min(), minmax=False)


def numpy_max(x):
    """
    Returns the maximum of an array.
    Deals with text as well.
    """
    return numpy_min_max(x, lambda x: x.max(), minmax=True)


def debug_dump(clname, obj, folder=None, ops=None):
    """
    Dumps an object for debug purpose.

    @param      clname  class name
    @param      obj     object
    @param      folder  folder
    @param      ops     operator to dump
    @return             filename
    """
    def debug_print_(obj, prefix=''):
        name = clname
        if isinstance(obj, dict):
            if 'in' in obj and 'out' in obj:
                nan_in = any(map(lambda o: any(map(numpy.isnan, o.ravel())),
                                 obj['in']))
                nan_out = any(map(lambda o: any(map(numpy.isnan, o.ravel())),
                                  obj['out']))
                if not nan_in and nan_out:
                    print("NAN-notin-out ", name, prefix,
                          {k: getattr(ops, k, '?') for k in getattr(ops, 'atts', {})})
                    return True
                return False  # pragma: no cover
            for k, v in obj.items():  # pragma: no cover
                debug_print_([v], k)
            return None  # pragma: no cover
        if isinstance(obj, list):
            for i, o in enumerate(obj):
                if o is None:
                    continue
                if any(map(numpy.isnan, o.ravel())):
                    print("NAN", prefix, i, name, o.shape)
            return None
        raise NotImplementedError(  # pragma: no cover
            "Unable to debug object of type {}.".format(type(obj)))

    dump = debug_print_(obj)
    if dump:
        name = 'cpu-{}-{}-{}.pkl'.format(
            clname, id(obj), id(ops))
        if folder is not None:
            name = "/".join([folder, name])
        with open(name, 'wb') as f:
            pickle.dump(obj, f)
        return name
    return None


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
                not isinstance(obj[0], dict)):  # pragma: no cover
            print("-='{}' list len={} min={} max={}".format(
                  k, len(obj), min(obj), max(obj)))
        else:  # pragma: no cover
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
        lines = [str(sig)]  # pragma: no cover
        for name in ['co_argcount', 'co_cellvars', 'co_code', 'co_consts', 'co_filename',
                     'co_firstlineno', 'co_flags', 'co_freevars', 'co_kwonlyargcount',
                     'co_lnotab', 'co_name', 'co_names', 'co_nlocals', 'co_stacksize',
                     'co_varnames']:  # pragma: no cover
            v = getattr(res.__code__, name, None)  # pylint: disable=E1101
            if v is not None:
                lines.append('%s=%r' % (name, v))
        raise RuntimeError(  # pragma: no cover
            "Defaults values of function '{}' (defaults={}) are missing.\nDefault: "
            "{}\n{}\n----\n{}".format(
                fct, res.__defaults__, defs, "\n".join(lines), code))  # pylint: disable=E1101
    return res
