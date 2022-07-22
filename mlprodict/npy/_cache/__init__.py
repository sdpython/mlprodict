"""
@file
@brief Cache documentation for OnnxOps.

.. versionadded:: 0.9
"""
import os


def cache_folder():
    """
    Returns this folder.
    """
    return os.path.abspath(os.path.dirname(__file__))
