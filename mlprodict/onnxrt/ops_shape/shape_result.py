"""
@file
@brief Class ShapeResult
"""
from enum import Enum

class OnnxKind(Enum):
    """
    Describes a result type.
    """    
    Tensor = 0
    Sequence = 0
    Map = 0


class ShapeResult:
    """
    Contains information about shape and type of a result
    in an onnx graph.

    :param shape: shape if the result is a tensor
    :param dtype: element type if the result is a tensor
    :param sparse: is a the tensor sparse
    :param mtype: kind of the result (see class @see cl OnnxKind)
    """
    def __init__(self, shape=None, dtype=None, sparse=False,
                 mtype=OnnxKind.Tensor):
        self.mtype = mtype
        self.shape = shape
        self.dtype = dtype
        self.sparse = sparse
        
    def __repr__(self):
        """
        Usual
        """
        return "%s(%r, %r, %r, %r)" % (
            self.__class__.__name__, self.shape, self.dtype,
            self.sparse, self.mtype)

