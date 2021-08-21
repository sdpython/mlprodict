"""
@file
@brief Templates to export an ONNX graph in a way it can we created again
with a python script.

.. versionadded:: 0.7
"""
import os
from textwrap import dedent
from functools import cache


@cache
def _get_file(name):
    """
    Retrieves one template.
    """
    this = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(this, "_onnx_export_templates_%s.tmpl" % name)
    if not os.path.exists(filename):
        raise FileNotFoundError(  # pragma: no cover
            "Unable to find template %r in folder %r." % (name, this))
    with open(filename, "r", encoding="utf-8") as f:
        return dedent(f.read())


def get_onnx_template():
    """
    Template to export :epkg:`ONNX` into :epkg:`onnx` code.
    """
    return _get_file('onnx')


def get_tf2onnx_template():
    """
    Template to export :epkg:`ONNX` into :epkg:`tensorflow-onnx` code.
    """
    return _get_file('tf2onnx')


def get_numpy_template():
    """
    Template to export :epkg:`ONNX` into :epkg:`numpy` code.
    """
    return _get_file('numpy')
