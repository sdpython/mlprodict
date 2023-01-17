"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Any, Optional, Tuple, Union
import numpy


class ElemTypeCst:
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


class ElemType(ElemTypeCst):
    """
    Allowed element type based on numpy dtypes.

    :param dtype: integer or a string
    """

    names_int = {
        att: getattr(ElemTypeCst, att)
        for att in dir(ElemTypeCst)
        if isinstance(getattr(ElemTypeCst, att), int)
    }

    int_names = {
        getattr(ElemTypeCst, att): att
        for att in dir(ElemTypeCst)
        if isinstance(getattr(ElemTypeCst, att), int)
    }

    numpy_map = {
        getattr(numpy, att): getattr(ElemTypeCst, att)
        for att in dir(ElemTypeCst)
        if isinstance(getattr(ElemTypeCst, att), int) and hasattr(numpy, att)
    }

    allowed = set(range(1, 17))

    ints = (
        ElemTypeCst.int8,
        ElemTypeCst.int16,
        ElemTypeCst.int32,
        ElemTypeCst.int64,
        ElemTypeCst.uint8,
        ElemTypeCst.uint16,
        ElemTypeCst.uint32,
        ElemTypeCst.uint64,
    )

    floats = (
        ElemTypeCst.float16,
        ElemTypeCst.bfloat16,
        ElemTypeCst.float32,
        ElemTypeCst.float64,
    )

    numeric = (
        ElemTypeCst.int8,
        ElemTypeCst.int16,
        ElemTypeCst.int32,
        ElemTypeCst.int64,
        ElemTypeCst.uint8,
        ElemTypeCst.uint16,
        ElemTypeCst.uint32,
        ElemTypeCst.uint64,
        ElemTypeCst.float16,
        ElemTypeCst.bfloat16,
        ElemTypeCst.float32,
        ElemTypeCst.float64,
    )

    __slots__ = ['dtype']

    def __init__(self, dtype: Union[str, int]):
        if isinstance(dtype, str):
            dtype = ElemType.names_int[dtype]
        elif dtype in ElemType.numpy_map:
            dtype = ElemType.numpy_map[dtype]
        elif dtype not in ElemType.allowed:
            raise ValueError(
                f"Unexpected dtype {dtype} not in {ElemType.allowed}.")
        self.dtype = dtype

    def __repr__(self) -> str:
        "usual"
        s = ElemType.int_names[self.dtype]
        return f"{self.__class__.__name__}(ElemType.{s})"


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

    def __init__(self, dtypes: Tuple[ElemType], shape: Optional[Tuple[int, ...]] = None, name: str = ""):
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
