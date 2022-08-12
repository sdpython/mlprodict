# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def stft(x, fft_length, hop_length, n_frames, window, onesided=False):
    """
    Applies one dimensional FFT with window weights.
    torch defines the number of frames as:
    `n_frames = 1 + (len - n_fft) / hop_length`.
    """
    one = op.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
    mtwo = op.Constant(value=make_tensor('mtwo', TensorProto.INT64, [1], [-2]))
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    last_axis = op.Sub(op.Shape(op.Shape(x)), one)
    axis = op.Constant(value=make_tensor('axis', TensorProto.INT64, [1], [-2]))
    axis2 = op.Constant(value=make_tensor('axis2', TensorProto.INT64, [1], [-3]))
    window_size = op.Shape(window)

    # building frames
    seq = op.SequenceEmpty(dtype=TensorProto.FLOAT)
    nf = op.Squeeze(n_frames, zero)
    for fs in range(nf):
        fs64 = op.Cast(fs, to=7)
        begin = op.Mul(fs64, hop_length)
        end = op.Add(begin, window_size)
        sliced_x = op.Slice(x, begin, end, axis)

        # sliced_x may be smaller
        new_dim = op.Shape(sliced_x, start=-2, end=-1)
        missing = op.Sub(window_size, new_dim)
        new_shape = op.Concat(
            op.Shape(sliced_x, start=0, end=-2),
            missing,
            op.Shape(sliced_x, start=-1),
            axis=0)
        cst = op.ConstantOfShape(
            new_shape, value=make_tensor('zerof', TensorProto.FLOAT, [1], [0]))
        pad_sliced_x = op.Concat(sliced_x, op.Cast(cst, to=1), axis=-2)

        # same size
        un_sliced_x = op.Unsqueeze(pad_sliced_x, axis2)
        seq = op.SequenceInsert(seq, un_sliced_x)

    # concatenation
    new_x = op.ConcatFromSequence(seq, axis=-3, new_axis=0)

    # calling weighted dft with weights=window
    shape_x = op.Shape(new_x)
    shape_x_short = op.Slice(shape_x, zero, mtwo, zero)
    shape_x_short_one = op.Add(op.Mul(shape_x_short, zero), one)
    window_shape = op.Concat(shape_x_short_one, window_size, one, axis=0)
    weights = op.Reshape(window, window_shape)
    weighted_new_x = op.Mul(new_x, weights)

    result = dft(weighted_new_x, fft_length, last_axis, onesided, False)

    # final transpose -3, -2
    two = op.Constant(value=make_tensor('two', TensorProto.INT64, [1], [2]))
    three  = op.Constant(value=make_tensor('three', TensorProto.INT64, [1], [3]))
    dim = op.Shape(op.Shape(result))
    ax1 = op.Sub(dim, three)
    ax2 = op.Sub(dim, two)
    return switch_axes(result, ax1, ax2)


def istft(x, fft_length, hop_length, window, onesided=False):
    """
    Reverses of `stft`.
    """
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    one = op.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
    two = op.Constant(value=make_tensor('two', TensorProto.INT64, [1], [2]))
    mone = op.Constant(value=make_tensor('mone', TensorProto.INT64, [1], [-1]))
    wone = op.Cast(op.ConstantOfShape(
        op.Shape(window),
        value=make_tensor('onef', TensorProto.FLOAT, [1], [1])), to=1)    
    axisf = op.Constant(value=make_tensor('axis3', TensorProto.INT64, [1], [-2]))
    n_frames = op.Shape(x, start=-2, end=-1)
    expected_signal_len = op.Add(fft_length, op.Mul(hop_length, op.Sub(n_frames, one)))

    # building frames
    seqr = op.SequenceEmpty()
    seqi = op.SequenceEmpty()
    seqc = op.SequenceEmpty()
    nf = op.Squeeze(n_frames, zero)
    for fs in range(nf):
        fs64 = op.Cast(fs, to=7)
        begin = fs64
        end = op.Add(fs64, one)
        frame_x = op.Squeeze(op.Slice(x, begin, end, axisf), axisf)

        # ifft
        ift = dft(frame_x, fft_length, mone, onesided, True)
        n_dims = op.Shape(op.Shape(ift))

        # real part
        n_dims_1 = op.Sub(n_dims, one)
        ytmp = op.Squeeze(op.Slice(ift, zero, one, n_dims_1), n_dims_1)
        ctmp = op.Mul(op.Cast(
            op.ConstantOfShape(
                op.Shape(ytmp),
                value=make_tensor('onef', TensorProto.FLOAT, [1], [1])),
            to=1), window)

        shape_begin = op.Shape(ytmp, end=-1)
        n_left = op.Mul(fs64, hop_length)
        size = op.Shape(ytmp, start=-1)
        n_right = op.Sub(expected_signal_len, op.Add(n_left, size))

        left_shape = op.Concat(shape_begin, n_left, axis=0)
        right_shape = op.Concat(shape_begin, n_right, axis=0)
        right = op.Cast(
            op.ConstantOfShape(
                right_shape,
                value=make_tensor('zerof', TensorProto.FLOAT, [1], [0])),
            to=1)
        left = op.Cast(
            op.ConstantOfShape(
                left_shape,
                value=make_tensor('zerof', TensorProto.FLOAT, [1], [0])),
            to=1)

        y = op.Concat(left, ytmp, right, axis=-1)
        yc = op.Concat(left, ctmp, right, axis=-1)

        # imaginary part
        itmp = op.Squeeze(op.Slice(ift, one, two, n_dims_1), n_dims_1)
        yi = op.Concat(left, itmp, right, axis=-1)

        # append
        seqr = op.SequenceInsert(seqr, op.Unsqueeze(y, mone))
        seqi = op.SequenceInsert(seqi, op.Unsqueeze(yi, mone))
        seqc = op.SequenceInsert(seqc, op.Unsqueeze(yc, mone))

    # concatenation
    redr = op.ConcatFromSequence(seqr, axis=-1, new_axis=0)
    redi = op.ConcatFromSequence(seqi, axis=-1, new_axis=0)
    redc = op.ConcatFromSequence(seqc, axis=-1, new_axis=0)

    # unweight
    resr = op.ReduceSum(redr, mone, keepdims=0)
    resi = op.ReduceSum(redi, mone, keepdims=0)
    resc = op.ReduceSum(redc, mone, keepdims=0)
    rr = op.Div(resr, resc)
    ri = op.Div(resi, resc)

    # Make complex
    rr0 = op.Unsqueeze(rr, zero)
    ri0 = op.Unsqueeze(ri, zero)
    conc = op.Concat(rr0, ri0, axis=0)

    # rotation, bring first dimension to the last position
    result_shape = op.Shape(conc)
    shape_cpl = op.Constant(value=make_tensor('shape_cpl', TensorProto.INT64, [2], [2, -1]))
    reshaped_result = op.Reshape(conc, shape_cpl)
    transposed = op.Transpose(reshaped_result, perm=[1, 0])
    other_dimensions = op.Slice(result_shape, one, op.Shape(result_shape), zero)
    final_shape = op.Concat(other_dimensions, two, axis=0)
    final = op.Reshape(transposed, final_shape)
    return final


class STFT(OpRun):

    atts = {'onesided': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=STFT.atts,
                       **options)

    def _run(self, signal, frame_step, window=None, frame_length=None,
             attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        raise NotImplementedError("not yet")
