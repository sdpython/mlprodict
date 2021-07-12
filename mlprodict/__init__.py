# -*- encoding: utf-8 -*-
"""
@file
@brief Ways to speed up predictions for a machine learned model.
"""

__version__ = "0.6.1447"
__author__ = "Xavier Dupré"


def check(log=False):
    """
    Checks the library is working.
    It raises an exception.
    If you want to disable the logs:

    @param      log     if True, display information, otherwise
    @return             0 or exception
    """
    return True


def _setup_hook(use_print=False):
    """
    If this function is added to the module,
    the help automation and unit tests call it first before
    anything goes on as an initialization step.
    """
    if use_print:
        print("Success: _setup_hook")


def load_ipython_extension(ip):  # pragma: no cover
    """
    To allow the call ``%load_ext mlprodict``

    @param      ip      from ``get_ipython()``
    """
    from .onnxrt.doc.nb_helper import register_onnx_magics as freg
    freg(ip)
