# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class OptionalGetElement(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if not isinstance(x, list):
            raise TypeError(  # pragma: no cover
                f"Unexpected type {type(x)!r} for x.")
        if len(x) > 0:
            return (x[0], )
        return ([], )


class OptionalHasElement(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if not isinstance(x, list):
            raise TypeError(  # pragma: no cover
                f"Unexpected type {type(x)!r} for x.")
        if len(x) > 0:
            return (numpy.array([e is not None for e in x]), )
        return ([], )
