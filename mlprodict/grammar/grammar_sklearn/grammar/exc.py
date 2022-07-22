"""
@file
@brief Exception definition.
"""


class Float32InfError(Exception):
    """
    Raised when a float is out of range and cannot be
    converted into a float32.
    """
    pass
