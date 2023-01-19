"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Any, Optional, Tuple, Union
import numpy


class ElemTypeCstInner:
    """
    Defines all possible types and tensor element type.
    """

    __slots__ = []

    bool_ = 9
    int8 = 3
    int16 = 5
    int32 = 6
    int64 = 7
    uint8 = 2
    uint16 = 4
    uint32 = 12
    uint64 = 13
    bfloat16 = 16
    float16 = 10
    float32 = 1
    float64 = 11
    complex64 = 14
    complex128 = 15


class ElemTypeCstSet(ElemTypeCstInner):
    """
    Sets of element types.
    """

    allowed = set(range(1, 17))

    ints = (
        ElemTypeCstInner.int8,
        ElemTypeCstInner.int16,
        ElemTypeCstInner.int32,
        ElemTypeCstInner.int64,
        ElemTypeCstInner.uint8,
        ElemTypeCstInner.uint16,
        ElemTypeCstInner.uint32,
        ElemTypeCstInner.uint64,
    )

    floats = (
        ElemTypeCstInner.float16,
        ElemTypeCstInner.bfloat16,
        ElemTypeCstInner.float32,
        ElemTypeCstInner.float64,
    )

    numerics = (
        ElemTypeCstInner.int8,
        ElemTypeCstInner.int16,
        ElemTypeCstInner.int32,
        ElemTypeCstInner.int64,
        ElemTypeCstInner.uint8,
        ElemTypeCstInner.uint16,
        ElemTypeCstInner.uint32,
        ElemTypeCstInner.uint64,
        ElemTypeCstInner.float16,
        ElemTypeCstInner.bfloat16,
        ElemTypeCstInner.float32,
        ElemTypeCstInner.float64,
    )

    @staticmethod
    def combined(type_set):
        s = 0
        for dt in type_set:
            s += 1 << dt
        return s


class ElemTypeCst(ElemTypeCstSet):
    """
    Combination of element types.
    """

    Bool = 1 << ElemTypeCstInner.bool_
    Int8 = 1 << ElemTypeCstInner.int8
    Int16 = 1 << ElemTypeCstInner.int16
    Int32 = 1 << ElemTypeCstInner.int32
    Int64 = 1 << ElemTypeCstInner.int64
    UInt8 = 1 << ElemTypeCstInner.uint8
    UInt16 = 1 << ElemTypeCstInner.uint16
    UInt32 = 1 << ElemTypeCstInner.uint32
    UInt64 = 1 << ElemTypeCstInner.uint64
    BFloat16 = 1 << ElemTypeCstInner.bfloat16
    Float16 = 1 << ElemTypeCstInner.float16
    Float32 = 1 << ElemTypeCstInner.float32
    Float64 = 1 << ElemTypeCstInner.float64
    Complex64 = 1 << ElemTypeCstInner.complex64
    Complex128 = 1 << ElemTypeCstInner.complex128

    Numerics = ElemTypeCstSet.combined(ElemTypeCstSet.numerics)
    Floats = ElemTypeCstSet.combined(ElemTypeCstSet.floats)
    Ints = ElemTypeCstSet.combined(ElemTypeCstSet.ints)


class ElemType(ElemTypeCst):
    """
    Allowed element type based on numpy dtypes.

    :param dtype: integer or a string
    """

    names_int = {
        att: getattr(ElemTypeCstInner, att)
        for att in dir(ElemTypeCstInner)
        if isinstance(getattr(ElemTypeCstInner, att), int)
    }

    int_names = {
        getattr(ElemTypeCstInner, att): att
        for att in dir(ElemTypeCstInner)
        if isinstance(getattr(ElemTypeCstInner, att), int)
    }

    set_names = {
        getattr(ElemTypeCst, att): att
        for att in dir(ElemTypeCst)
        if isinstance(getattr(ElemTypeCst, att), int) and "A" <= att[0] <= "Z"
    }

    numpy_map = {
        getattr(numpy, att): getattr(ElemTypeCst, att)
        for att in dir(ElemTypeCst)
        if isinstance(getattr(ElemTypeCst, att), int) and hasattr(numpy, att)
    }

    __slots__ = ['dtype']

    def __init__(self, dtype: Union[str, int]):
        if isinstance(dtype, str):
            dtype = ElemType.names_int[dtype]
        elif dtype in ElemType.numpy_map:
            dtype = ElemType.numpy_map[dtype]
        elif dtype not in ElemType.allowed:
            raise ValueError(
                f"Unexpected dtype {dtype} not in {ElemType.allowed}.")
        self.dtype: int = dtype

    def __repr__(self) -> str:
        "usual"
        s = ElemType.int_names[self.dtype]
        return f"{self.__class__.__name__}(ElemType.{s})"


class Par:
    """
    Defines a parameter type.

    :param dtype: parameter type
    :param optional: is optional or not
    """

    __slots__ = ["dtype", "optional"]

    map_names = {int: "int", float: "float", str: "str"}

    @classmethod
    def __class_getitem__(cls, dtype):
        return Par(dtype)

    def __init__(self, dtype: type, optional: bool = False):
        self.dtype = dtype
        self.optional = optional

    def __repr__(self) -> str:
        "usual"
        if self.optional:
            return f"{self.__class__.__name__}({Par.map_names[self.dtype]}, optional=True)"
        return f"{self.__class__.__name__}({Par.map_names[self.dtype]})"

    def __str__(self) -> str:
        "usual"
        if self.optional:
            return f"{self.__class__.__name__}{Par.map_names[self.dtype]}, optional=True)"
        return f"{self.__class__.__name__}[{Par.map_names[self.dtype]}]"


class OptPar(Par):
    """
    Defines an optional parameter type.

    :param dtype: parameter type
    """

    @classmethod
    def __class_getitem__(cls, dtype):
        return OptPar(dtype)

    def __init__(self, dtype):
        Par.__init__(self, dtype, True)

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}({Par.map_names[self.dtype]})"

    def __str__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}[{Par.map_names[self.dtype]}]"


class TensorType:
    """
    Used to annotate functions.

    :param dtypes: tuple of :class:`ElemType`
    :param shape: tuple of integer or strings or None
    :param name: name of the type
    """

    __slots__ = ['dtypes', 'shape', 'name']

    @classmethod
    def __class_getitem__(cls, dtypes):
        return TensorType(dtypes)

    def __init__(self, dtypes: Tuple[ElemType],
                 shape: Optional[Union[int, Tuple[int, ...]]] = None,
                 name: str = ""):
        if isinstance(dtypes, ElemType):
            dtypes = (dtypes,)
        elif (isinstance(dtypes, str) or dtypes in ElemType.allowed or
              dtypes in ElemType.numpy_map):
            dtypes = (ElemType(dtypes), )
        if not isinstance(dtypes, tuple):
            raise TypeError(f"dtypes must be a tuple not {type(dtypes)}.")
        check = []
        for dt in dtypes:
            if isinstance(dt, ElemType):
                check.append(dt)
            elif dt in ElemType.allowed:
                check.append(ElemType(dt))
            else:
                raise TypeError(f"Unexpected type {type(dt)} in {dtypes}.")
        self.dtypes = tuple(check)
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.name = name

    def __repr__(self) -> str:
        "usual"
        if len(self.dtypes) == 1:
            st = repr(self.dtypes[0])
        else:
            st = repr(self.dtypes)
        if self.shape:
            if self.name:
                return f"{self.__class__.__name__}({st}, {self.shape!r}, {self.name!r})"
            return f"{self.__class__.__name__}({st}, {self.shape!r})"
        if self.name:
            return f"{self.__class__.__name__}({st}, {self.name!r})"
        return f"{self.__class__.__name__}({st})"

    def _name_set(self):
        s = 0
        for dt in self.dtypes:
            s += 1 << dt.dtype
        try:
            return ElemType.set_names[s]
        except KeyError:
            raise RuntimeError(
                f"Unable to guess element type name for {s}: "
                f"{repr(self)} in {ElemType.set_names}.")

    def __str__(self) -> str:
        """
        Simplified display.
        """
        name = self._name_set()
        if self.shape:
            sh = str(self.shape).strip("()").replace(" ", "")
            sig = f"{name}[{sh}]"
        else:
            sig = f"{name}[]"
        if self.name:
            return f"{sig}({self.name})"
        return sig

    def issuperset(self, tensor_type: "TensorType") -> bool:
        """
        Tells if *self* is a superset of *tensor_type*.
        """
        set1 = set(t.dtype for t in self.dtypes)
        set2 = set(t.dtype for t in tensor_type.dtypes)
        if not set1.issuperset(set2):
            return False
        if self.shape is None:
            return True
        if tensor_type.shape is None:
            return False
        if len(self.shape) != len(tensor_type.shape):
            return False
        for a, b in zip(self.shape, tensor_type.shape):
            if isinstance(a, int):
                if a != b:
                    return False
        return True


class Float32:
    """
    For simpler annotation.
    """
    @classmethod
    def __class_getitem__(cls, shape):
        return TensorType(ElemType.float32, shape=shape)


class Float64:
    """
    For simpler annotation.
    """
    @classmethod
    def __class_getitem__(cls, shape):
        return TensorType(ElemType.float64, shape=shape)


class Int64:
    """
    For simpler annotation.
    """
    @classmethod
    def __class_getitem__(cls, shape):
        return TensorType(ElemType.int64, shape=shape)
