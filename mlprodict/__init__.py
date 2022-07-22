# -*- encoding: utf-8 -*-
"""
@file
@brief Python runtime for ONNX and others tools to help
converting investigate issues with ONNX models.
"""

__version__ = "0.8.1858"
__author__ = "Xavier Dupré"
__max_supported_opset__ = 15  # Converters are tested up to this version.
__max_supported_opsets__ = {
    '': __max_supported_opset__,
    'ai.onnx.ml': 2}
# Converters are tested up to this version.
__max_supported_opset_experimental__ = 16
__max_supported_opsets_experimental__ = {
    '': __max_supported_opset_experimental__,
    'ai.onnx.ml': 3}


def get_ir_version(opv):
    """
    Returns the corresponding `IR_VERSION` based on the selected opset.
    See :epkg:`ONNX Version`.

    :param opv: opset
    :return: runtime version
    """
    if isinstance(opv, dict):
        opv = opv['']
    if opv is None or opv >= 15:
        return 8
    if opv >= 12:
        return 7
    if opv >= 11:  # pragma no cover
        return 6
    if opv >= 10:  # pragma no cover
        return 5
    if opv >= 9:  # pragma no cover
        return 4
    if opv >= 8:  # pragma no cover
        return 4
    return 3  # pragma no cover


def check(log=False):
    """
    Checks the library is working.
    It raises an exception.
    If you want to disable the logs:

    @param      log     if True, display information, otherwise
    @return             0 or exception
    """
    return True


def load_ipython_extension(ip):  # pragma: no cover
    """
    To allow the call ``%load_ext mlprodict``

    @param      ip      from ``get_ipython()``
    """
    from .nb_helper import register_onnx_magics as freg
    freg(ip)
