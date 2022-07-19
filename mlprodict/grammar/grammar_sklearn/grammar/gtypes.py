"""
@file
@brief Types definition.
"""
import numpy
from .api_extension import AutoType


class MLType(AutoType):
    """
    Base class for every type.
    """

    def validate(self, value):
        """
        Checks that the value is of this type.
        """
        # It must be overwritten.
        self._cache = value

    def cast(self, value):
        """
        Converts *value* into this type.
        """
        raise NotImplementedError()  # pragma: no cover


class MLNumType(MLType):
    """
    Base class for numerical types.
    """

    def _format_value_json(self, value, hook=None):
        return str(value)

    def _format_value_c(self, value, hook=None):
        return str(value)

    def _copy_c(self, src, dst, hook=None):
        if hook == "typeref":
            return f"*{dst} = {src};"
        return f"{dst} = {src};"


class MLNumTypeSingle(MLNumType):
    """
    int32 or float32
    """

    def __init__(self, numpy_type, name, ctype, key):
        self.numpy_type = numpy_type
        self.name = name
        self.ctype = ctype
        self.key = key

    @property
    def CTypeSingle(self):
        """
        Returns *ctype*.
        """
        return self.ctype

    def validate(self, value):
        """
        Checks that the value is of this type.
        """
        MLNumType.validate(self, value)
        if not isinstance(value, self.numpy_type):
            raise TypeError(  # pragma: no cover
                f"'{type(value)}' is not a {self.numpy_type}.")
        return value

    def cast(self, value):
        """
        Exports *value* into this type.
        """
        if isinstance(value, numpy.float32):
            raise TypeError(  # pragma: no cover
                f"No need to cast, already a {self.numpy_type}")
        if isinstance(value, numpy.ndarray):
            if len(value) != 1:
                raise ValueError(  # pragma: no cover
                    f"Dimension of array must be one single {self.numpy_type}")
            return value[0]
        raise NotImplementedError(  # pragma: no cover
            "Unable to cast '{0}' into a {0}".format(type(self.numpy_type)))

    def softcast(self, value):
        """
        Exports *value* into this type, does it anyway without verification.
        """
        if isinstance(value, numpy.ndarray):
            v = value.ravel()
            if len(v) != 1:
                raise ValueError(  # pragma: no cover
                    f"Cannot cast shape {value.shape} into {self.numpy_type}")
            return self.numpy_type(v[0])
        return self.numpy_type(value)

    def _export_common_c(self, ctype, hook=None, result_name=None):
        if hook == 'type':
            return {'code': ctype} if result_name is None else {'code': ctype + ' ' + result_name}
        if result_name is None:
            return {'code': ctype}
        return {'code': ctype + ' ' + result_name, 'result_name': result_name}

    def _byref_c(self):
        return "&"

    def _export_json(self, hook=None, result_name=None):
        return 'float32'

    def _export_c(self, hook=None, result_name=None):
        if hook == 'typeref':
            return {'code': self.ctype + '*'} if result_name is None else {'code': self.ctype + '* ' + result_name}
        return self._export_common_c(self.ctype, hook, result_name)

    def _format_value_json(self, value, hook=None):
        if hook is None or self.key not in hook:
            return value
        return hook[self.key](value)

    def _format_value_c(self, value, hook=None):
        if hook is None or self.key not in hook:
            return f"({self.ctype}){value}"
        return hook[self.key](value)


class MLNumTypeFloat32(MLNumTypeSingle):
    """
    A numpy.float32.
    """

    def __init__(self):
        MLNumTypeSingle.__init__(
            self, numpy.float32, 'float32', 'float', 'float32')


class MLNumTypeFloat64(MLNumTypeSingle):
    """
    A numpy.float64.
    """

    def __init__(self):
        MLNumTypeSingle.__init__(
            self, numpy.float64, 'float64', 'double', 'float64')


class MLNumTypeInt32(MLNumTypeSingle):
    """
    A numpy.int32.
    """

    def __init__(self):
        MLNumTypeSingle.__init__(self, numpy.int32, 'int32', 'int', 'int32')


class MLNumTypeInt64(MLNumTypeSingle):
    """
    A numpy.int64.
    """

    def __init__(self):
        MLNumTypeSingle.__init__(
            self, numpy.int32, 'int64', 'int64_t', 'int64')


class MLNumTypeBool(MLNumTypeSingle):
    """
    A numpy.bool.
    """

    def __init__(self):
        MLNumTypeSingle.__init__(self, numpy.bool_, 'BL', 'bool', 'bool')


class MLTensor(MLType):
    """
    Defines a tensor with a dimension and a single type for what it contains.
    """

    def __init__(self, element_type, dim):
        if not isinstance(element_type, MLType):
            raise TypeError(  # pragma: no cover
                f'element_type must be of MLType not {type(element_type)}')
        if not isinstance(dim, tuple):
            raise TypeError(  # pragma: no cover
                'dim must be a tuple.')
        if len(dim) == 0:
            raise ValueError(  # pragma: no cover
                "dimension must not be null.")
        for d in dim:
            if d == 0:
                raise ValueError(  # pragma: no cover
                    "No dimension can be null.")
        self.dim = dim
        self.element_type = element_type

    @property
    def CTypeSingle(self):
        """
        Returns *ctype*.
        """
        return self.element_type.ctype

    def validate(self, value):
        """
        Checks that the value is of this type.
        """
        MLType.validate(self, value)
        if not isinstance(value, numpy.ndarray):
            raise TypeError(  # pragma: no cover
                f"value is not a numpy.array but '{type(value)}'")
        if self.dim != value.shape:
            raise ValueError(  # pragma: no cover
                f"Dimensions do not match {self.dim}={value.shape}")
        rvalue = value.ravel()
        for i, num in enumerate(rvalue):
            try:
                self.element_type.validate(num)
            except TypeError as e:  # pragma: no cover
                raise TypeError(
                    f'Unable to convert an array due to value index {i}: {num}') from e
        return value

    def _byref_c(self):
        return ""

    def _format_value_json(self, value, hook=None):
        if hook is None or 'array' not in hook:
            return value
        return hook['array'](value)

    def _format_value_c(self, value, hook=None):
        return f"{{{', '.join(self.element_type._format_value_c(x) for x in value)}}}"

    def _export_json(self, hook=None, result_name=None):
        return f'{self.element_type._export_json(hook=hook)}:{self.dim}'

    def _export_c(self, hook=None, result_name=None):
        if len(self.dim) != 1:
            raise NotImplementedError(  # pragma: no cover
                'Only 1D vector implemented.')
        if hook is None:
            raise ValueError(  # pragma: no cover
                "hook must contains either 'signature' or 'declare'.")
        if hook == 'signature':
            if result_name is None:
                raise ValueError(  # pragma: no cover
                    "result_name must be specified.")
            return {'code': "{0}[{1}] {2}".format(self.element_type._export_c(hook=hook)['code'],
                                                  self.dim[0], result_name),
                    'result_name': result_name}
        elif hook == 'declare':
            if result_name is None:
                raise ValueError(  # pragma: no cover
                    "result_name must be specified.")
            dc = self.element_type._export_c(
                hook=hook, result_name=result_name)
            return {'code': f"{dc['code']}[{self.dim[0]}]"}
        elif hook == 'type':
            return {'code': f"{self.element_type._export_c(hook=hook)['code']}*"}
        elif hook == 'typeref':
            if result_name is None:
                return {'code': f"{self.element_type._export_c(hook='type')['code']}*"}
            code = self.element_type._export_c(hook='type')['code']
            return {'code': f"{code}* {result_name}", 'result_name': result_name}
        else:
            raise ValueError(  # pragma: no cover
                f"hook must contains either 'signature' or 'declare' not '{hook}'.")

    def _copy_c(self, src, dest, hook=None):
        if len(self.dim) != 1:
            raise NotImplementedError(  # pragma: no cover
                'Only 1D vector implemented.')
        code = self.element_type._export_c(hook='type')['code']
        return f"memcpy({dest}, {src}, {self.dim[0]}*sizeof({code}));"
