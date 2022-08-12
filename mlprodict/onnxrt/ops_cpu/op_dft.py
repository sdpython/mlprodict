# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def _fft(x, fft_length, axis):
    if fft_length is None:
        fft_length = [x.shape[axis]]
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


def _cfft(x, fft_length, axis, onesided=False, normalize=False):
    # if normalize:
    #    raise NotImplementedError()
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
    res = _fft(c, fft_length, axis=axis)
    if onesided:
        slices = [slice(0, a) for a in res.shape]
        slices[axis] = slice(0, res.shape[axis] // 2 + 1)
        return res[tuple(slices)]
    return res


def _ifft(x, fft_length, axis=-1, onesided=False):
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
    if onesided:
        slices = [slice() for a in res.shape]
        slices[axis] = slice(0, res.shape[axis] // 2 + 1)
        return res[tuple(slices)]
    return tr


def _cifft(x, fft_length, axis=-1, onesided=False):
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
    return _ifft(c, fft_length, axis=axis, onesided=onesided)


class DFT(OpRun):

    atts = {'axis': 1, 'inverse': 0, 'onesided': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=DFT.atts,
                       **options)

    def _run(self, x, dft_length=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if dft_length is None:
            dft_length = numpy.array([x.shape[self.axis]], dtype=numpy.int64)
        if self.inverse:
            res = _cifft(x, dft_length, axis=self.axis, onesided=self.onesided)
        else:
            res = _cfft(x, dft_length, axis=self.axis, onesided=self.onesided)
        return (res.astype(x.dtype), )
