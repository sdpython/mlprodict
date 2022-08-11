# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class DFT(OpRun):

    atts = {'axis': 1, 'inverse': 0, 'onesided': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=DFT.atts,
                       **options)

    @staticmethod
    def _fft(x, fft_length, axis):
        ft = numpy.fft.fft(x, fft_length[0], axis=axis)
        r = numpy.real(ft)
        i = numpy.imag(ft)
        merged = numpy.vstack([r[numpy.newaxis, ...], i[numpy.newaxis, ...]])
        perm = numpy.arange(len(merged.shape))
        perm[:-1] = perm[1:]
        perm[-1] = 0
        tr = numpy.transpose(merged, list(perm))
        if tr.shape[-1] != 2:
            raise RuntimeError(
                f"Unexpected shape {tr.shape}, x.shape={x.shape} "
                f"fft_length={fft_length}.")
        return tr

    @staticmethod
    def _cfft(x, fft_length, axis):
        if x.shape[-1] == 1:
            tmp = x
        else:
            slices = [slice(0, x) for x in x.shape]
            slices[-1] = slice(0, x.shape[-1], 2)
            real = x[tuple(slices)]
            slices[-1] = slice(1, x.shape[-1], 2)
            imag = x[tuple(slices)]
            tmp = real + 1j * imag
        c = numpy.squeeze(tmp, -1)
        return DFT._fft(c, fft_length, axis=axis)

    @staticmethod
    def _cifft(x, fft_length, axis=-1):
        if x.shape[-1] == 1:
            tmp = x
        else:
            slices = [slice(0, x) for x in x.shape]
            slices[-1] = slice(0, x.shape[-1], 2)
            real = x[tuple(slices)]
            slices[-1] = slice(1, x.shape[-1], 2)
            imag = x[tuple(slices)]
            tmp = real + 1j * imag
        c = numpy.squeeze(tmp, -1)
        return DFT._ifft(c, fft_length, axis=axis)

    @staticmethod
    def _ifft(x, fft_length, axis=-1):
        ft = numpy.fft.ifft(x, fft_length[0], axis=axis)
        r = numpy.real(ft)
        i = numpy.imag(ft)
        merged = numpy.vstack([r[numpy.newaxis, ...], i[numpy.newaxis, ...]])
        perm = numpy.arange(len(merged.shape))
        perm[:-1] = perm[1:]
        perm[-1] = 0
        tr = numpy.transpose(merged, list(perm))
        if tr.shape[-1] != 2:
            raise RuntimeError(
                f"Unexpected shape {tr.shape}, x.shape={x.shape} "
                f"fft_length={fft_length}.")
        return tr

    def _run(self, x, dft_length=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if dft_length is None:
            dft_length = numpy.array([x.shape[self.axis]], dtype=numpy.int64)
        if self.inverse:
            res = DFT._cifft(x, dft_length, axis=self.axis)
        else:
            if self.onesided:
                raise NotImplementedError("not yet")
            else:
                res = DFT._cfft(x, dft_length, axis=self.axis)
        return (res.astype(x.dtype), )
