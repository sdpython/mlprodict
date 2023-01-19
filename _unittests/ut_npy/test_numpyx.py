"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from onnx.reference import ReferenceEvaluator
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.numpyx import ElemType, TensorType
from mlprodict.npy.numpyx_types import Float32, Float64, Int64
from mlprodict.npy.numpyx_core import Input, Var
from mlprodict.npy.numpyx_functions import absolute


class TestNumpyx(ExtTestCase):

    def test_tensor(self):
        dt = TensorType("float32")
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEmpty(dt.shape)
        self.assertEqual(repr(dt), "TensorType(ElemType(ElemType.float32))")
        dt = TensorType["float32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(repr(dt), "TensorType(ElemType(ElemType.float32))")
        dt = TensorType[numpy.float32]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(repr(dt), "TensorType(ElemType(ElemType.float32))")
        self.assertEmpty(dt.shape)

        self.assertRaise(lambda: TensorType([]), TypeError)
        self.assertRaise(lambda: TensorType(numpy.str_), TypeError)
        self.assertRaise(lambda: TensorType(
            (numpy.float32, numpy.str_)), TypeError)

    def test_sig(self):

        def local1(x: TensorType[ElemType.floats]) -> TensorType[ElemType.floats]:
            return x

        def local2(x: TensorType(ElemType.floats, name="T")) -> TensorType(ElemType.floats, name="T"):
            return x

        def local3(x: Float32["N", 1]) -> Float32["N", 1]:
            return x

        def local4(x: Float64["N", 1]) -> Int64["N", 1]:
            return x

    def test_numpy_add(self):
        f = absolute(Input())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx()
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'x': x})
        self.assertEqualArray(y, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
