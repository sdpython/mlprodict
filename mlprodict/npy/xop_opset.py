# pylint: disable=E0602
"""
@file
@brief Xop API to build onnx graphs. Inspired from :epkg:`sklearn-onnx`.

.. versionadded:: 0.9
"""
import numpy
from .xop import loadop


def OnnxReduceSumApi11(*x, axes=None, keepdims=1, op_version=None,
                       output_names=None):
    """
    Adds operator ReduceSum with opset>=13 following API from opset 12.
    """
    if op_version is None:
        raise RuntimeError(  # pragma: no cover
            "op_version must be specified.")
    if op_version is None or op_version >= 13:
        OnnxReduceSum = loadop('ReduceSum')
        if axes is None:
            return OnnxReduceSum(
                *x, keepdims=keepdims, op_version=op_version,
                output_names=output_names)
        return OnnxReduceSum(
            *x, numpy.array(axes, dtype=numpy.int64),
            keepdims=keepdims, op_version=op_version,
            output_names=output_names)
    if op_version >= 11:
        OnnxReduceSum_11 = loadop('ReduceSum_11')
        if axes is None:
            return OnnxReduceSum_11(
                *x, keepdims=keepdims,
                op_version=op_version, output_names=output_names)
        return OnnxReduceSum_11(
            *x, axes=axes, keepdims=keepdims,
            op_version=op_version, output_names=output_names)
    OnnxReduceSum_1 = loadop('ReduceSum_1')
    if axes is None:
        return OnnxReduceSum_1(*x, keepdims=keepdims,
                               op_version=op_version,
                               output_names=output_names)
    return OnnxReduceSum_1(*x, axes=axes, keepdims=keepdims,
                           op_version=op_version, output_names=output_names)


def OnnxSplitApi18(*x, axis=0, split=None, num_outputs=None,
                   op_version=None, output_names=None):
    """
    Adds operator Split with opset>=13 following API from opset 11.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 18:
        OnnxSplit_18 = loadop('Split_18')
        if split is None:
            if num_outputs is None:
                if output_names is None:
                    raise RuntimeError(
                        "split or num_outputs or output_names "
                        "must be specified since opset 18.")
                num_outputs = len(output_names)
            if num_outputs is None:
                raise AttributeError(
                    "num_outputs cannot be None for Split-18.")
            return OnnxSplit_18(  # noqa
                *x, axis=axis, op_version=op_version,
                num_outputs=num_outputs, output_names=output_names)
        if num_outputs is None:
            return OnnxSplit_18(  # noqa
                *x, numpy.array(split, dtype=numpy.int64), axis=axis,
                op_version=op_version, output_names=output_names)
        return OnnxSplit_18(  # noqa
            *x, numpy.array(split, dtype=numpy.int64), axis=axis,
            num_outputs=num_outputs, op_version=op_version,
            output_names=output_names)
    if op_version >= 13:
        OnnxSplit_13 = loadop('Split_13')
        if split is None:
            return OnnxSplit_13(  # noqa
                *x, axis=axis, op_version=op_version,
                output_names=output_names)
        return OnnxSplit_13(  # noqa
            *x, numpy.array(split, dtype=numpy.int64), axis=axis,
            op_version=op_version, output_names=output_names)
    if op_version >= 11:
        OnnxSplit_11 = loadop('Split_11')
        if split is None:
            return OnnxSplit_11(  # noqa
                *x, axis=axis, op_version=op_version,
                output_names=output_names)
        return OnnxSplit_11(  # noqa
            *x, split=split, axis=axis, op_version=op_version,
            output_names=output_names)
    OnnxSplit_2 = loadop('Split_2')
    if split is None:
        return OnnxSplit_2(  # noqa
            *x, axis=axis, op_version=op_version, output_names=output_names)
    return OnnxSplit_2(*x, split=split, axis=axis,  # noqa
                       op_version=op_version, output_names=output_names)


def OnnxSqueezeApi11(*x, axes=None, op_version=None,
                     output_names=None):
    """
    Adds operator Squeeze with opset>=13 following API from opset 11.
    """
    if op_version is None:
        raise RuntimeError(  # pragma: no cover
            "op_version must be specified.")
    if op_version is None or op_version >= 13:
        OnnxSqueeze = loadop('Squeeze')
        return OnnxSqueeze(
            *x, numpy.array(axes, dtype=numpy.int64),
            op_version=op_version, output_names=output_names)
    if op_version >= 11:
        OnnxSqueeze_11 = loadop('Squeeze_11')
        return OnnxSqueeze_11(
            *x, axes=axes, op_version=op_version,
            output_names=output_names)
    OnnxSqueeze_1 = loadop('Squeeze_1')
    return OnnxSqueeze_1(*x, axes=axes,
                         op_version=op_version, output_names=output_names)


def OnnxUnsqueezeApi11(*x, axes=None, op_version=None,
                       output_names=None):
    """
    Adds operator Unsqueeze with opset>=13 following API from opset 11.
    """
    if op_version is None:
        raise RuntimeError(  # pragma: no cover
            "op_version must be specified.")
    if op_version is None or op_version >= 13:
        OnnxUnsqueeze = loadop('Unsqueeze')
        return OnnxUnsqueeze(
            *x, numpy.array(axes, dtype=numpy.int64),
            op_version=op_version, output_names=output_names)
    if op_version >= 11:
        OnnxUnsqueeze_11 = loadop('Unsqueeze_11')
        return OnnxUnsqueeze_11(
            *x, axes=axes, op_version=op_version,
            output_names=output_names)
    OnnxUnsqueeze_1 = loadop('Unsqueeze_1')
    return OnnxUnsqueeze_1(*x, axes=axes,
                           op_version=op_version, output_names=output_names)


def OnnxReshapeApi13(*x, allowzero=0, op_version=None,
                     output_names=None):
    """
    Adds operator Reshape with opset>=14 following API from opset 13.
    """
    if op_version is None:
        raise RuntimeError(  # pragma: no cover
            "op_version must be specified.")
    if op_version is None or op_version >= 14:
        OnnxReshape = loadop('Reshape')
        return OnnxReshape(
            *x, allowzero=allowzero,
            op_version=op_version, output_names=output_names)
    if op_version >= 13:
        OnnxReshape_13 = loadop('Reshape_13')
        return OnnxReshape_13(
            *x, op_version=op_version, output_names=output_names)
    OnnxReshape_5 = loadop('Reshape_5')
    return OnnxReshape_5(
        *x, op_version=op_version, output_names=output_names)


def OnnxReduceAnyApi18(cl18, cl13, cl11, cl1, *x, axes=None, keepdims=1,
                       op_version=None, output_names=None):
    """
    Adds operator Reduce* with opset>=18 following API from opset 17.
    """
    if op_version is None or op_version >= 18:
        if axes is None:
            return cl18(
                *x, keepdims=keepdims, op_version=op_version,
                output_names=output_names)
        return cl18(
            *x, numpy.array(axes, dtype=numpy.int64),
            keepdims=keepdims, op_version=op_version,
            output_names=output_names)
    if op_version >= 13:
        if axes is None:
            return cl13(*x, keepdims=keepdims,
                        op_version=op_version,
                        output_names=output_names)
        return cl13(*x, axes=axes, keepdims=keepdims,
                    op_version=op_version, output_names=output_names)
    if op_version >= 11:
        if axes is None:
            return cl11(*x, keepdims=keepdims,
                        op_version=op_version,
                        output_names=output_names)
        return cl11(*x, axes=axes, keepdims=keepdims,
                    op_version=op_version, output_names=output_names)
    if axes is None:
        return cl1(*x, keepdims=keepdims,
                   op_version=op_version,
                   output_names=output_names)
    return cl1(*x, axes=axes, keepdims=keepdims,
               op_version=op_version, output_names=output_names)


def OnnxReduceSumSquareApi18(*x, axes=None, keepdims=1, op_version=None,
                             output_names=None):
    """
    Adds operator ReduceSumSquare with opset>=18 following API from opset 17.
    """
    OnnxReduceSumSquare = loadop('ReduceSumSquare')
    (OnnxReduceSumSquare_13, OnnxReduceSumSquare_11,
     OnnxReduceSumSquare_1) = loadop(
        'ReduceSumSquare_13', 'ReduceSumSquare_11', 'ReduceSumSquare_1')
    return OnnxReduceAnyApi18(
        OnnxReduceSumSquare, OnnxReduceSumSquare_13,
        OnnxReduceSumSquare_11, OnnxReduceSumSquare_1,
        *x, axes=axes, keepdims=keepdims, op_version=op_version,
        output_names=output_names)


def OnnxReduceMeanApi18(*x, axes=None, keepdims=1, op_version=None,
                        output_names=None):
    """
    Adds operator ReduceMean with opset>=18 following API from opset 17.
    """
    OnnxReduceMean = loadop('ReduceMean')
    (OnnxReduceMean_13, OnnxReduceMean_11, OnnxReduceMean_1) = loadop(
        'ReduceMean_13', 'ReduceMean_11', 'ReduceMean_1')
    return OnnxReduceAnyApi18(
        OnnxReduceMean, OnnxReduceMean_13,
        OnnxReduceMean_11, OnnxReduceMean_1,
        *x, axes=axes, keepdims=keepdims, op_version=op_version,
        output_names=output_names)


def OnnxReduceL218(*x, axes=None, keepdims=1, op_version=None,
                   output_names=None):
    """
    Adds operator ReduceMean with opset>=18 following API from opset 17.
    """
    OnnxReduceL2 = loadop('ReduceL2')
    (OnnxReduceL2_13, OnnxReduceL2_11, OnnxReduceL2_1) = loadop(
        'ReduceL2_13', 'ReduceL2_11', 'ReduceL2_1')
    return OnnxReduceAnyApi18(
        OnnxReduceL2, OnnxReduceL2_13,
        OnnxReduceL2_11, OnnxReduceL2_1,
        *x, axes=axes, keepdims=keepdims, op_version=op_version,
        output_names=output_names)


def OnnxReduceL2_typed(dtype, x, axes=None, keepdims=1, op_version=None,
                       output_names=None):
    """
    Adds operator ReduceL2 for float or double.
    """
    OnnxMul, OnnxSqrt = loadop('Mul', 'Sqrt')
    if dtype == numpy.float32:
        return OnnxReduceL218(
            x, axes=axes, keepdims=keepdims,
            op_version=op_version, output_names=output_names)
    x2 = OnnxMul(x, x, op_version=op_version)
    red = OnnxReduceSumApi11(
        x2, axes=[1], keepdims=1, op_version=op_version)
    return OnnxSqrt(
        red, op_version=op_version, output_names=output_names)
