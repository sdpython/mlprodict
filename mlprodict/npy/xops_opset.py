# pylint: disable=E0602
"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import numpy as np


def OnnxReduceSumApi11(*x, axes=None, keepdims=1, op_version=None,
                       output_names=None):
    """
    Adds operator ReduceSum with opset>=13 following API from opset 12.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 13:
        if axes is None:
            return OnnxReduceSum(
                *x, keepdims=keepdims, op_version=op_version,
                output_names=output_names)
        return OnnxReduceSum(
            *x, np.array(axes, dtype=np.int64),
            keepdims=keepdims, op_version=op_version,
            output_names=output_names)
    if op_version >= 11:
        if axes is None:
            return OnnxReduceSum_11(
                *x, keepdims=keepdims,
                op_version=op_version, output_names=output_names)
        return OnnxReduceSum_11(
            *x, axes=axes, keepdims=keepdims,
            op_version=op_version, output_names=output_names)
    if axes is None:
        return OnnxReduceSum_1(*x, keepdims=keepdims,
                               op_version=op_version,
                               output_names=output_names)
    return OnnxReduceSum_1(*x, axes=axes, keepdims=keepdims,
                           op_version=op_version, output_names=output_names)


def OnnxSplitApi11(*x, axis=0, split=None, op_version=None,
                   output_names=None):
    """
    Adds operator Split with opset>=13 following API from opset 11.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 13:
        if split is None:
            return OnnxSplit(
                *x, axis=axis, op_version=op_version,
                output_names=output_names)
        return OnnxSplit(
            *x, np.array(split, dtype=np.int64), axis=axis,
            op_version=op_version, output_names=output_names)
    if op_version >= 11:
        if split is None:
            return OnnxSplit_11(
                *x, axis=axis, op_version=op_version,
                output_names=output_names)
        return OnnxSplit_11(
            *x, split=split, axis=axis, op_version=op_version,
            output_names=output_names)
    if split is None:
        return OnnxSplit_2(
            *x, axis=axis, op_version=op_version, output_names=output_names)
    return OnnxSplit_2(*x, split=split, axis=axis,
                       op_version=op_version, output_names=output_names)


def OnnxSqueezeApi11(*x, axes=None, op_version=None,
                     output_names=None):
    """
    Adds operator Squeeze with opset>=13 following API from opset 11.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 13:
        return OnnxSqueeze(
            *x, np.array(axes, dtype=np.int64),
            op_version=op_version, output_names=output_names)
    if op_version >= 11:
        return OnnxSqueeze_11(
            *x, axes=axes, op_version=op_version,
            output_names=output_names)
    return OnnxSqueeze_1(*x, axes=axes,
                         op_version=op_version, output_names=output_names)


def OnnxUnsqueezeApi11(*x, axes=None, op_version=None,
                       output_names=None):
    """
    Adds operator Unsqueeze with opset>=13 following API from opset 11.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 13:
        return OnnxUnsqueeze(
            *x, np.array(axes, dtype=np.int64),
            op_version=op_version, output_names=output_names)
    if op_version >= 11:
        return OnnxUnsqueeze_11(
            *x, axes=axes, op_version=op_version,
            output_names=output_names)
    return OnnxUnsqueeze_1(*x, axes=axes,
                           op_version=op_version, output_names=output_names)


def OnnxReduceL2_typed(dtype, x, axes=None, keepdims=1, op_version=None,
                       output_names=None):
    """
    Adds operator ReduceL2 for float or double.
    """
    if dtype == np.float32:
        return OnnxReduceL2(
            x, axes=axes, keepdims=keepdims,
            op_version=op_version, output_names=output_names)
    x2 = OnnxMul(x, x, op_version=op_version)
    red = OnnxReduceSumApi11(
        x2, axes=[1], keepdims=1, op_version=op_version)
    return OnnxSqrt(
        red, op_version=op_version, output_names=output_names)


def OnnxReshapeApi13(*x, allowzero=0, op_version=None,
                     output_names=None):
    """
    Adds operator Reshape with opset>=14 following API from opset 13.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 14:
        return OnnxReshape(
            *x, allowzero=allowzero,
            op_version=op_version, output_names=output_names)
    if op_version >= 13:
        return OnnxReshape_13(
            *x, op_version=op_version, output_names=output_names)
    return OnnxReshape_5(
        *x, op_version=op_version, output_names=output_names)
