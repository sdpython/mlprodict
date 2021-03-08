"""
@file
@brief :epkg:`numpy` annotations.

.. versionadded:: 0.6
"""
import inspect
from collections import OrderedDict
from typing import TypeVar, Generic
import numpy

try:
    numpy_bool = numpy.bool_
except AttributeError:  # pragma: no cover
    numpy_bool = bool

try:
    numpy_str = numpy.str
except AttributeError:  # pragma: no cover
    numpy_str = str

Shape = TypeVar("Shape")
DType = TypeVar("DType")


all_dtypes = (numpy.float32, numpy.float64,
              numpy.int32, numpy.int64,
              numpy.uint32, numpy.uint64)


def get_args_kwargs(fct, n_optional):
    """
    Extracts arguments and optional parameters of a function.

    :param fct: function
    :param n_optional: number of arguments to consider as
        optional arguments and not parameters, this parameter skips
        the first *n_optional* paramerters
    :return: arguments, OrderedDict
    """
    params = inspect.signature(fct).parameters
    if n_optional == 0:
        items = list(params.items())
        args = [name for name, p in params.items()
                if p.default == inspect.Parameter.empty]
    else:
        items = []
        args = []
        for name, p in params.items():
            if p.default == inspect.Parameter.empty:
                args.append(name)
            else:
                if n_optional > 0:
                    args.append(name)
                    n_optional -= 1
                else:
                    items.append((name, p))

    kwargs = OrderedDict((name, p.default) for name, p in items
                         if (p.default != inspect.Parameter.empty and
                             name != 'op_version'))
    return args, kwargs


class NDArray(numpy.ndarray, Generic[Shape, DType]):
    """
    Used to annotation ONNX numpy functions.

    .. versionadded:: 0.6
    """
    class ShapeType:
        "Stores shape information."

        def __init__(self, params):
            self.__args__ = params

    def __class_getitem__(cls, params):
        "Overwrites this method."
        if not isinstance(params, tuple):
            params = (params,)  # pragma: no cover
        return NDArray.ShapeType(params)


class _NDArrayAlias:
    """
    Ancestor to custom signature.

    :param dtypes: input dtypes
    :param dtypes_out: output dtypes
    :param n_optional: number of optional parameters, 0 by default
    :param nvars: True if the function allows an infinite number of inputs,
        this is incompatible with parameter *n_optional*.

    .. versionadded:: 0.6
    """

    def __init__(self, dtypes=None, dtypes_out=None, n_optional=None,
                 nvars=False):
        "constructor"
        if dtypes is None:
            raise ValueError("dtypes cannot be None.")  # pragma: no cover
        if isinstance(dtypes, tuple) and len(dtypes) == 0:
            raise TypeError("dtypes must not be empty.")  # pragma: no cover
        if isinstance(dtypes, tuple) and not isinstance(dtypes[0], tuple):
            dtypes = tuple(t if isinstance(t, str) else (t,) for t in dtypes)
        if isinstance(dtypes, str) and '_' in dtypes:
            dtypes, dtypes_out = dtypes.split('_')
        if not isinstance(dtypes, (tuple, list)):
            dtypes = (dtypes, )

        self.mapped_types = {}
        self.dtypes = _NDArrayAlias._process_type(
            dtypes, self.mapped_types, 0)
        if dtypes_out is None:
            self.dtypes_out = (self.dtypes[0], )
        elif isinstance(dtypes_out, int):
            self.dtypes_out = (self.dtypes[dtypes_out], )
        else:
            if not isinstance(dtypes_out, (tuple, list)):
                dtypes_out = (dtypes_out, )
            self.dtypes_out = _NDArrayAlias._process_type(
                dtypes_out, self.mapped_types, 0)
        self.n_optional = 0 if n_optional is None else n_optional
        self.n_variables = nvars

        if not isinstance(self.dtypes, tuple):
            raise TypeError(  # pragma: no cover
                "self.dtypes must be a tuple not {}.".format(self.dtypes))
        if (len(self.dtypes) == 0 or
                not isinstance(self.dtypes[0], tuple)):
            raise TypeError(  # pragma: no cover
                "Type mismatch in self.dtypes: {}.".format(self.dtypes))
        if (len(self.dtypes[0]) == 0 or
                isinstance(self.dtypes[0][0], tuple)):
            raise TypeError(  # pragma: no cover
                "Type mismatch in self.dtypes: {}.".format(self.dtypes))

        if not isinstance(self.dtypes_out, tuple):
            raise TypeError(  # pragma: no cover
                "self.dtypes_out must be a tuple not {}.".format(self.dtypes_out))
        if (len(self.dtypes_out) == 0 or
                not isinstance(self.dtypes_out[0], tuple)):
            raise TypeError(  # pragma: no cover
                "Type mismatch in self.dtypes_out: {}.".format(self.dtypes_out))
        if (len(self.dtypes_out[0]) == 0 or
                isinstance(self.dtypes_out[0][0], tuple)):
            raise TypeError(  # pragma: no cover
                "Type mismatch in self.dtypes_out: {}.".format(self.dtypes_out))

        if self.n_variables and self.n_optional > 0:
            raise RuntimeError(  # pragma: no cover
                "n_variables and n_optional cannot be positive at "
                "the same type.")

    @staticmethod
    def _process_type(dtypes, mapped_types, index):
        """
        Nicknames such as `floats`, `int`, `ints`, `all`
        can be used to describe multiple inputs for
        a signature. This function intreprets that.

        .. runpython::
            :showcode:

            from mlprodict.npy.onnx_numpy_annotation import _NDArrayAlias
            for name in ['all', 'int', 'ints', 'floats', 'T']:
                print(name, _NDArrayAlias._process_type(name, {'T': 0}, 0))
        """
        if isinstance(dtypes, str):
            if ":" in dtypes:
                name, dtypes = dtypes.split(':')
                if name in mapped_types and dtypes != mapped_types[name]:
                    raise RuntimeError(
                        "Type name mismatch for '%s:%s' in %r." % (
                            name, dtypes, list(sorted(mapped_types))))
                mapped_types[name] = (dtypes, index)
            if dtypes == "all":
                dtypes = all_dtypes
            elif dtypes == "int":
                dtypes = (numpy.int64, )
            elif dtypes == "bool":
                dtypes = (numpy_bool, )
            elif dtypes == "floats":
                dtypes = (numpy.float32, numpy.float64)
            elif dtypes == "ints":
                dtypes = (numpy.int32, numpy.int64)
            elif dtypes not in mapped_types:
                raise ValueError(  # pragma: no cover
                    "Unexpected shortcut for dtype %r." % dtypes)
            elif not isinstance(dtypes, tuple):
                dtypes = (dtypes, )
            return dtypes

        if isinstance(dtypes, (tuple, list)):
            insig = [_NDArrayAlias._process_type(dt, mapped_types, index + d)
                     for d, dt in enumerate(dtypes)]
            return tuple(insig)

        if dtypes in all_dtypes:
            return dtypes

        raise NotImplementedError(
            "Unexpected input dtype %r." % dtypes)

    def __repr__(self):
        "usual"
        return "%s(%r, %r, %r)" % (
            self.__class__.__name__, self.dtypes, self.dtypes_out,
            self.n_optional)

    def _to_onnx_dtype(self, dtype, shape):
        from skl2onnx.common.data_types import _guess_numpy_type
        if dtype == numpy.bool_:
            dtype = numpy.bool
        return _guess_numpy_type(dtype, shape)

    def _get_output_types(self, key):
        """
        Tries to infer output types.
        """
        res = []
        for i, o in enumerate(self.dtypes_out):
            if not isinstance(o, tuple):
                raise TypeError(  # pragma: no cover
                    "All outputs must be tuple, output %d is %r."
                    "" % (i, o))
            if (len(o) == 1 and (o[0] in all_dtypes or
                                 o[0] in (bool, numpy_bool, str, numpy_str))):
                res.append(o[0])
            elif len(o) == 1 and o[0] in self.mapped_types:
                info = self.mapped_types[o[0]]
                res.append(key[info[1]])
            elif key[0] in o:
                res.append(key[0])
            else:
                raise RuntimeError(  # pragma: no cover
                    "Unable to guess output type for output %d, "
                    "input types are %r, expected output is %r."
                    "" % (i, key, o))
        return tuple(res)

    def get_inputs_outputs(self, args, kwargs, version):
        """
        Returns the list of inputs, outputs.

        :param args: list of arguments
        :param kwargs: list of optional arguments
        :param version: required version
        :return: *tuple(inputs, outputs, n_input_range)*,
            each of them is a list of tuple with the name and the dtype
        """
        for k, v in kwargs.items():
            if isinstance(v, type):
                raise RuntimeError(  # pragma: no cover
                    "Default value for argument %r must not be of type %r"
                    "." % (k, v))

        def _possible_names():
            yield 'y'
            yield 'z'  # pragma: no cover
            yield 'o'  # pragma: no cover
            for i in range(0, 10000):  # pragma: no cover
                yield 'o%d' % i

        key = version if isinstance(version, tuple) else (version, )
        items = list(kwargs.items())

        if self.n_variables:
            ngiven = len(key) - len(items)
            key_types = key[:ngiven]
            names = tuple("x%d" % i for i in range(ngiven))
            args = list(names)
        else:
            names = tuple(args)
            kw = 0
            while len(names) < len(self.dtypes):
                names = names + (items[kw][0], )
                kw += 1
            kwargs = OrderedDict(items[kw:])
            args = list(names)
            key_types = key[:len(args)] if len(key) > len(args) else key

        li = len(key_types)
        optional = self.n_optional - (len(args) - len(key_types))
        for i in range(0, optional):
            li = len(key_types) - i - 1
            if li < 0:
                break
            if key_types[li] is not None:
                break
        key_types = key_types[:li + 1]

        onnx_types = [self._to_onnx_dtype(k, None) for k in key_types]
        inputs = list(zip(args, onnx_types))

        key_out = self._get_output_types(key)
        onnx_types_out = [self._to_onnx_dtype(k, None) for k in key_out]

        names_out = []
        names_in = set(inp[0] for inp in inputs)
        for _ in key_out:
            for name in _possible_names():
                if name not in names_in:
                    name_out = name
                    break
            names_out.append(name_out)
            names_in.add(name_out)

        outputs = list(zip(names_out, onnx_types_out))
        optional = self.n_optional - (len(args) - len(inputs))
        if optional < 0:
            raise RuntimeError(
                "optional cannot be negative %r (self.n_optional=%r, "
                "len(self.dtypes)=%r, len(inputs)=%r)." % (
                    optional, self.n_optional, len(self.dtypes),
                    len(inputs)))
        return inputs, kwargs, outputs, optional

    def shape_calculator(self, dims):
        """
        Returns expected dimensions given the input dimensions.
        """
        if len(dims) == 0:
            return None
        res = [dims[0]]
        for _ in dims[1:]:
            res.append(None)
        return res


class NDArrayType(_NDArrayAlias):
    """
    Shortcut to simplify signature description.

    :param dtypes: input dtypes
    :param dtypes_out: output dtypes
    :param n_optional: number of optional parameters, 0 by default
    :param nvars: True if the function allows an infinite number of inputs,
        this is incompatible with parameter *n_optional*.

    .. versionadded:: 0.6
    """

    def __init__(self, dtypes=None, dtypes_out=None, n_optional=None, nvars=False):
        _NDArrayAlias.__init__(self, dtypes=dtypes, dtypes_out=dtypes_out,
                               n_optional=n_optional, nvars=nvars)


class NDArrayTypeSameShape(NDArrayType):
    """
    Shortcut to simplify signature description.

    :param dtypes: input dtypes
    :param dtypes_out: output dtypes
    :param n_optional: number of optional parameters, 0 by default
    :param nvars: True if the function allows an infinite number of inputs,
        this is incompatible with parameter *n_optional*.

    .. versionadded:: 0.6
    """

    def __init__(self, dtypes=None, dtypes_out=None, n_optional=None, nvars=False):
        NDArrayType.__init__(self, dtypes=dtypes, dtypes_out=dtypes_out,
                             n_optional=n_optional, nvars=nvars)


class NDArraySameType(NDArrayType):
    """
    Shortcut to simplify signature description.

    :param dtypes: input dtypes

    .. versionadded:: 0.6
    """

    def __init__(self, dtypes=None):
        if dtypes is None:
            raise ValueError("dtypes cannot be None.")  # pragma: no cover
        if isinstance(dtypes, str) and "_" in dtypes:
            raise ValueError(  # pragma: no cover
                "dtypes cannot include '_' meaning two different types.")
        if isinstance(dtypes, tuple):
            raise ValueError(  # pragma: no cover
                "dtypes must be a single type.")
        NDArrayType.__init__(self, dtypes=dtypes)

    def __repr__(self):
        "usual"
        return "%s(%r)" % (
            self.__class__.__name__, self.dtypes)


class NDArraySameTypeSameShape(NDArraySameType):
    """
    Shortcut to simplify signature description.

    :param dtypes: input dtypes

    .. versionadded:: 0.6
    """

    def __init__(self, dtypes=None):
        NDArraySameType.__init__(self, dtypes=dtypes)
