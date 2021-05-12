"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from onnx import TensorProto
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument  # pylint: disable=E0611
from pyquickhelper.pycode import ExtTestCase, skipif_appveyor
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnx_tools.optim.onnx_helper import change_input_first_dimension
from mlprodict.onnx_tools.onnx2py_helper import (
    to_bytes, from_bytes, numpy_max, numpy_min, _type_to_string,
    _numpy_array)
from mlprodict.onnxrt.ops_cpu._op_helper import proto2dtype
from mlprodict.onnxrt import OnnxInference


class TestOnnxHelper(ExtTestCase):

    def common_test(self, data):
        pb = to_bytes(data)
        self.assertIsInstance(pb, bytes)
        data2 = from_bytes(pb)
        self.assertEqualArray(data, data2)

    def test_conversion_float(self):
        data = numpy.array([[0, 1], [2, 3], [4, 5]], dtype=numpy.float32)
        self.common_test(data)

    def test_conversion_double(self):
        data = numpy.array([[0, 1], [2, 3], [4, 5]], dtype=numpy.float64)
        self.common_test(data)

    def test_conversion_int64(self):
        data = numpy.array([[0, 1], [2, 3], [4, 5]], dtype=numpy.int64)
        self.common_test(data)

    @skipif_appveyor("unstable")
    def test_change_input_first_dimension(self):
        iris = load_iris()
        X, _ = iris.data, iris.target
        clr = KMeans()
        clr.fit(X)

        model_onnx = to_onnx(clr, X.astype(numpy.float32))
        oinf0 = OnnxInference(model_onnx, runtime='onnxruntime1')

        for inp in model_onnx.graph.input:
            dim = inp.type.tensor_type.shape.dim[0].dim_value
            self.assertEqual(dim, 0)
        new_model = change_input_first_dimension(model_onnx, 2)
        for inp in model_onnx.graph.input:
            dim = inp.type.tensor_type.shape.dim[0].dim_value
            self.assertEqual(dim, 0)
        for inp in new_model.graph.input:  # pylint: disable=E1101
            dim = inp.type.tensor_type.shape.dim[0].dim_value
            self.assertEqual(dim, 2)

        oinf = OnnxInference(new_model, runtime='onnxruntime1')
        self.assertRaise(lambda: oinf.run({'X': X.astype(numpy.float32)}),
                         InvalidArgument)
        res0 = oinf0.run({'X': X[:2].astype(numpy.float32)})
        res = oinf.run({'X': X[:2].astype(numpy.float32)})
        for k, v in res.items():
            self.assertEqual(v.shape[0], 2)
            self.assertEqualArray(res0[k], v)

        new_model = change_input_first_dimension(new_model, 0)
        oinf = OnnxInference(new_model, runtime='onnxruntime1')
        res = oinf.run({'X': X[:3].astype(numpy.float32)})
        for k, v in res.items():
            self.assertEqual(v.shape[0], 3)

    def test_proto2dtype(self):
        tt = [TensorProto.FLOAT, TensorProto.DOUBLE,  # pylint: disable=E1101
              TensorProto.BOOL, TensorProto.STRING,  # pylint: disable=E1101
              TensorProto.INT64, TensorProto.INT32]  # pylint: disable=E1101
        for t in tt:
            dt = proto2dtype(t)
            self.assertTrue(dt is not None)
        self.assertRaise(lambda: proto2dtype(671), ValueError)

    def test_numpy_min_max(self):
        val = numpy.array(['a', 'b'])
        r = numpy_min(val)
        self.assertEqual(r, "'a'")
        r = numpy_max(val)
        self.assertEqual(r, "'b'")
        val = numpy.array(['a', 0])
        r = numpy_min(val)
        self.assertEqual(r, "'0'")
        val = numpy.array(['aaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'b'])
        r = numpy_min(val)
        self.assertEqual(r, "'aaaaaaaaaa...'")
        val = numpy.array(['a', 'bbbbbbbbbbbbbbbbbbbbbbbbbbb'])
        r = numpy_max(val)
        self.assertEqual(r, "'bbbbbbbbbb...'")

    def test__type_to_string(self):
        d = dict(a=0, b=1)
        self.assertRaise(lambda: _type_to_string(d), KeyError)

    def test__numpy_array(self):
        v = numpy.array([0])
        f = _numpy_array(v)
        self.assertEqualArray(f, v)


if __name__ == "__main__":
    unittest.main()
