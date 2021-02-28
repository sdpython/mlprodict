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
except AttributeError:
    numpy_bool = bool

try:
    numpy_str = numpy.str
except AttributeError:
    numpy_str = str

Shape = TypeVar("Shape")
DType = TypeVar("DType")


all_dtypes = (numpy.float32, numpy.float64,
              numpy.int32, numpy.int64,
              numpy.uint32, numpy.uint64)


def get_args_kwargs(fct):
    """
    Extracts arguments and optional parameters of a function.

    :param fct: function
    :return: arguments, OrderedDict
    """
    params = inspect.signature(fct).parameters
    args = [name for name, p in params.items()
            if p.default == inspect.Parameter.empty]
    kwargs = OrderedDict((name, p.default) for name, p in params.items()
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
            params = (params,)
        return NDArray.ShapeType(params)


class _NDArrayAlias:

    @staticmethod
    def _process_type(dtypes):
        """
        Nicknames such as `floats`, `int`, `ints`, `all`
        can be used to describe multiple inputs for
        a signature. This function intreprets that.

        .. runpython::
            :showcode:

            from mlprodict.npy.onnx_numpy_annotation import _NDArrayAlias
            for name in ['all', 'int', 'ints', 'floats']:
                print(name, _NDArrayAlias._process_type(name))
        """
        if isinstance(dtypes, str):
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
            else:
                raise ValueError(
                    "Unexpected shortcut for dtype %r." % dtypes)
            return dtypes

        if isinstance(dtypes, (tuple, list)):
            insig = [_NDArrayAlias._process_type(dt) for dt in dtypes]
            return tuple(insig)

        if dtypes in all_dtypes:
            return dtypes

        raise NotImplementedError(
            "Unexpected input dtype %r." % dtypes)

    def __init__(self, dtypes=None, dtypes_out=None, n_optional=None, nvars=False):
        if dtypes is None:
            raise ValueError("dtypes cannot be None.")
        if isinstance(dtypes, str) and '_' in dtypes:
            dtypes, dtypes_out = dtypes.split('_')
        if not isinstance(dtypes, (tuple, list)):
            dtypes = (dtypes, )
        self.dtypes = _NDArrayAlias._process_type(dtypes)
        if dtypes_out is None:
            self.dtypes_out = (self.dtypes[0], )
        else:
            if not isinstance(dtypes_out, (tuple, list)):
                dtypes_out = (dtypes_out, )
            self.dtypes_out = _NDArrayAlias._process_type(dtypes_out)
        self.n_optional = 0 if n_optional is None else n_optional
        self.n_variables = nvars

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
        k0 = key[0]
        res = []
        for i, o in enumerate(self.dtypes_out):
            if not isinstance(o, tuple):
                raise TypeError(
                    "All outputs must be tuple, output %d is %r."
                    "" % (i, o))
            if (len(o) == 1 and (o[0] in all_dtypes or
                                 o[0] in (bool, numpy_bool, str, numpy_str))):
                res.append(o[0])
            elif k0 in o:
                res.append(k0)
            else:
                raise RuntimeError(
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
                raise RuntimeError(
                    "Default value for argument %r must not be of type %r"
                    "." % (k, v))

        def _possible_names():
            yield 'y'
            yield 'z'
            yield 'o'
            for i in range(0, 10000):
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
        optional = self.n_optional
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
    :param nvars: True if the function allows a variable number of inputs

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
    :param nvars: True if the function allows a variable number of inputs

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
            raise ValueError("dtypes cannot be None.")
        if isinstance(dtypes, str) and "_" in dtypes:
            raise ValueError(
                "dtypes cannot include '_' meaning two different types.")
        if isinstance(dtypes, tuple):
            raise ValueError(
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
