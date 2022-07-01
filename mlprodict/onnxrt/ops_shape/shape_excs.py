"""
@file
@brief Errors and exceptions.
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
