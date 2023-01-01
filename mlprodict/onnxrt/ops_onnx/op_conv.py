# pylint: disable=W0221
"""
@file
@brief Extension for :epkg:`ReferenceEvaluator`.
"""
import numpy
from ..ops_cpu.op_conv_ import ConvFloat, ConvDouble  # pylint: disable=E0611,E0401
from ._op import OpRunExtended


class Conv(OpRunExtended):
    """
    C++ implementation of operator Conv for :epkg:`ReferenceEvaluator`.
    See following example.

    .. runpython::
        :showcode:

        import numpy
        from numpy.testing import assert_allclose
        from onnx import TensorProto
        from onnx.checker import check_model
        from onnx.helper import (
            make_graph, make_model, make_node,
            make_opsetid, make_tensor_value_info)
        from onnx.reference import ReferenceEvaluator
        from mlprodict.plotting.text_plot import onnx_simple_text_plot
        from mlprodict.onnxrt.ops_onnx.op_conv import Conv
        from cpyquickhelper.numbers import measure_time

        # creating a model
        X = make_tensor_value_info("X", TensorProto.FLOAT, [
                                   None, None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [
                                   None, None, None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [
                                   None, None, None, None])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3])
        node = make_node(
            "Conv", ["X", "W", "B"], ["Y"], pads=[1, 1, 1, 1],
            dilations=[1, 1], strides=[2, 2])
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        # prints the model
        print(onnx_simple_text_plot(onnx_model))

        # comparing without and with C++ implementation
        sess1 = ReferenceEvaluator(onnx_model)
        sess2 = ReferenceEvaluator(onnx_model, new_ops=[Conv])

        sH, sW = 224, 224
        X = numpy.random.randn(1, 1, sH, sW).astype(numpy.float32)
        W = numpy.random.randn(1, 1, 3, 3).astype(numpy.float32)
        B = numpy.array([[[[0]]]], dtype=numpy.float32)

        expected = sess1.run(None, {"X": X, "W": W, "B": B})[0]
        got = sess2.run(None, {"X": X, "W": W, "B": B})[0]

        # checking it is the same
        assert_allclose(expected, got, atol=1e-5)

        # comparing the time
        t1 = measure_time(
            lambda: sess1.run(None, {"X": X, "W": W, "B": B}),
            repeat=5, number=5, div_by_number=True)
        print("No C++:", t1["average"])
        t2 = measure_time(
            lambda: sess2.run(None, {"X": X, "W": W, "B": B}),
            repeat=5, number=5, div_by_number=True)
        print("With C++:", t2["average"])
        print("speedup:", t1["average"] / t2["average"])
    """

    def get_impl(self, dtype=None, auto_pad=None, dilations=None, group=None,
                 kernel_shape=None, pads=None, strides=None):
        """
        Instantiates the C++ implementation and caches it.
        """
        key = self.get_cache_key(
            auto_pad=auto_pad, dilations=dilations,
            group=group, kernel_shape=kernel_shape, pads=pads,
            strides=strides, dtype=dtype)
        if self.has_cache_key(key):
            return self.get_cache_impl(key)
        if dtype == numpy.float32:
            rt = ConvFloat()
        elif dtype == numpy.float64:
            rt = ConvDouble()
        else:
            raise RuntimeError(
                f"No C++ implementation for Conv is available for dtype={dtype}.")
        rt.init(auto_pad,
                numpy.array(dilations, dtype=numpy.int64),
                group,
                numpy.array(kernel_shape, dtype=numpy.int64),
                numpy.array(pads, dtype=numpy.int64),
                numpy.array(strides, dtype=numpy.int64))
        self.cache_impl(key, rt)
        return rt

    def _run(self, X, W, B=None, auto_pad=None, dilations=None, group=None,
             kernel_shape=None, pads=None, strides=None):
        if len(X.shape) < 3:
            raise ValueError(
                f"X must have at least 3 dimensions but its shape is {X.shape}.")
        if X is None:
            raise ValueError(  # pragma: no cover
                "X cannot be None for operator %r, ONNX=%r" % (
                    type(self), self.onnx_node))
        if min(X.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Unable to run operator Conv on an empty matrix. X.shape={X.shape!r}.")
        if min(W.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Unable to run operator Conv on an empty matrix. W.shape={W.shape!r}.")
        if B is not None and min(B.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Unable to run operator Conv on an empty matrix. B.shape={B.shape!r}.")
        rt = self.get_impl(dtype=X.dtype, auto_pad=auto_pad,
                           dilations=dilations, group=group,
                           kernel_shape=kernel_shape or W.shape[-2:],
                           pads=pads, strides=strides)
        return (rt.compute(X, W, B), )
