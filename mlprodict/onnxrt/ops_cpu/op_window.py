# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.onnx_pb import TensorProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from ._op import OpRun


class _CommonWindow:

    def _begin(self, size):
        if self.periodic == 1:
            N_1 = size
        else:
            N_1 = size - 1
        ni = numpy.arange(size, dtype=self.dtype)
        return ni, N_1

    def _end(self, size, res):
        return (res.astype(self.dtype), )


class BlackmanWindow(OpRun, _CommonWindow):
    """
    Returns
    :math:`\\omega_n = 0.42 - 0.5 \\cos \\left( \\frac{2\\pi n}{N-1} \\right) +
    0.08 \\cos \\left( \\frac{4\\pi n}{N-1} \\right)`
    where *N* is the window length.
    See `blackman_window
    <https://pytorch.org/docs/stable/generated/torch.blackman_window.html>`_
    """

    atts = {'output_datatype': TensorProto.FLOAT, 'periodic': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=BlackmanWindow.atts,
                       **options)
        self.dtype = TENSOR_TYPE_TO_NP_TYPE[self.output_datatype]

    def _run(self, size, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        # ni, N_1 = self._begin(size)
        ni, N_1 = numpy.arange(size, dtype=self.dtype), size
        alpha = 0.42
        beta = 0.08
        pi = 3.1415
        y = 0.42
        y -= numpy.cos((ni * (pi * 2)) / N_1) / 2
        y += numpy.cos((ni * (pi * 4)) / N_1) * beta
        return (self._end(size, y), )


class HannWindow(OpRun, _CommonWindow):
    """
    Returns
    :math:`\\omega_n = \\sin^2\\left( \\frac{\\pi n}{N-1} \\right)`
    where *N* is the window length.
    See `hann_window
    <https://pytorch.org/docs/stable/generated/torch.hann_window.html>`_
    """

    atts = {'output_datatype': TensorProto.FLOAT, 'periodic': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=HannWindow.atts,
                       **options)
        self.dtype = TENSOR_TYPE_TO_NP_TYPE[self.output_datatype]

    def _run(self, size, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        ni, N_1 = self._begin(size)
        res = numpy.sin(ni * 3.1415 / N_1) ** 2
        return self._end(size, res)


class HammingWindow(OpRun, _CommonWindow):
    """
    Returns
    :math:`\\omega_n = \\alpha - \\beta \\cos \\left( \\frac{\\pi n}{N-1} \\right)`
    where *N* is the window length.
    See `hamming_window
    <https://pytorch.org/docs/stable/generated/torch.hamming_window.html>`_.
    `alpha=0.54, beta=0.46`
    """

    atts = {'output_datatype': TensorProto.FLOAT, 'periodic': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=HammingWindow.atts,
                       **options)
        self.dtype = TENSOR_TYPE_TO_NP_TYPE[self.output_datatype]

    def _run(self, size, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        ni, N_1 = self._begin(size)
        alpha = 25. / 46.
        beta = 1 - alpha
        res = alpha - numpy.cos(ni * 3.1415 * 2 / N_1) * beta
        return self._end(size, res)
