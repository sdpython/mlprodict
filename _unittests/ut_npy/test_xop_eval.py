"""
@brief      test log(time=5s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=E0611
    InvalidArgument)
from mlprodict.npy.xop import loadop
from mlprodict.npy.xop_convert import OnnxSubOnnx
from mlprodict.onnxrt import OnnxInference


class TestXOpsEval(ExtTestCase):

    def test_onnx_abs(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        x = numpy.array([0, 1], dtype=numpy.float32)
        y = ov.f({'X': x})
        self.assertEqualArray(numpy.abs(x), y['Y'])
        y = ov.f(x)
        self.assertEqualArray(numpy.abs(x), y)
        ov = OnnxAbs('X')
        y = ov.f(x)
        self.assertEqualArray(numpy.abs(x), y)

    def test_onnx_abs_log(self):
        rows = []

        def myprint(*args):
            rows.extend(args)
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        x = numpy.array([0, 1], dtype=numpy.float32)
        ov.f({'X': x}, verbose=10, fLOG=myprint)
        self.assertStartsWith("[OnnxOperator.f] creating node 'Abs'", rows[0])

    def test_onnx_transpose(self):
        OnnxTranspose = loadop("Transpose")
        ov = OnnxTranspose('X', perm=[1, 0], output_names=['Y'])
        x = numpy.array([[0, 1]], dtype=numpy.float32)
        y = ov.f(x)
        self.assertEqualArray(x.T, y)

    def test_onnx_onnxruntime(self):
        OnnxTranspose = loadop("Transpose")
        ov = OnnxTranspose('X', perm=[1, 0], output_names=['Y'])
        x = numpy.array([[0, 1]], dtype=numpy.float32)
        try:
            y = ov.f(x, runtime='onnxruntime1')
        except (InvalidArgument, RuntimeError) as e:
            if 'Invalid tensor data type' in str(e):
                # output is undefined
                return
            raise e
        self.assertEqualArray(x.T, y)

    def test_onnx_abs_add(self):
        OnnxAbs, OnnxAdd = loadop("Abs", "Add")
        ov = OnnxAdd('X', OnnxAbs('X'), output_names=['Y'])
        x = numpy.array([0, 1], dtype=numpy.float32)
        y = ov.f({'X': x})
        self.assertEqualArray(numpy.abs(x) + x, y['Y'])
        y = ov.f(x)
        self.assertEqualArray(numpy.abs(x) + x, y)
        ov = OnnxAdd('X', OnnxAbs('X'), output_names=['Y'])
        y = ov.f(x)
        self.assertEqualArray(numpy.abs(x) + x, y)

    def test_onnx_abs_exc(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        x = numpy.array([0, 1], dtype=numpy.float32)
        self.assertRaise(lambda: ov.f())
        self.assertRaise(lambda: ov.f(x, x))

    def test_onnx_abs_subonnx(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)

        sub = OnnxSubOnnx(onx, 'X', output_names=['Y'])
        x = numpy.array([-2, 2], dtype=numpy.float32)
        y = sub.f(x)
        self.assertEqualArray(numpy.abs(x), y)

    def test_onnx_operator_item(self):
        X = numpy.array([[4, 5, 6], [7, 0, 1]], dtype=numpy.float32)
        W = numpy.array([[1, 0.5, 0.6], [0.5, 0.2, 0.3]], dtype=numpy.float32)

        OnnxReduceMean, OnnxTopK, OnnxGatherElements = loadop(
            'ReduceMean', 'TopK', 'GatherElements')

        topk = OnnxTopK('X', numpy.array([2], dtype=numpy.int64), axis=1)

        r2 = topk.f(X)
        r1 = topk.f({'X': X})
        self.assertEqualArray(r1['Indices1'], r2[1])
        self.assertEqualArray(r1['Values0'], r2[0])

        dist = OnnxGatherElements('W', topk[1], axis=1)

        names = dist.find_named_inputs()
        self.assertEqual(['W', 'X'], names)
        r1 = dist.f({'X': X, 'W': W})
        r2 = dist.f(W, X)
        self.assertEqualArray(r1['output0'], r2)

        result = OnnxReduceMean(dist * topk[0], axes=[1])
        onx = result.to_onnx(numpy.float32, numpy.float32)

        sess = OnnxInference(onx)
        name = sess.output_names[0]
        res = sess.run({'X': X, 'W': W})
        res2 = result.f({'X': X, 'W': W})
        self.assertEqualArray(res[name], res2['reduced0'])


if __name__ == "__main__":
    # TestXOpsEval().test_onnx_operator_item()
    unittest.main(verbosity=2)
