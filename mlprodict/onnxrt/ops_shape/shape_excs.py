"""
@file
@brief Errors and exceptions for @see cl OnnxShapeInference.
"""


class ShapeInferenceException(RuntimeError):
    """
    Raised when shape inference fails.
    """
    pass


class ShapeInferenceMissing(RuntimeError):
    """
    Raised when an operator is missing.
    """
    pass


class NotImplementedShapeInferenceError(NotImplementedError):
    """
    Shape Inference can be implemented but is currently not.
    """
    pass


class ShapeInferenceDimensionError(RuntimeError):
    """
    Raised when the shape cannot continue
    due to unknown dimension.
    """
    pass
