"""
@brief      test tree node (time=4s)
"""
import os
import unittest
import re
import numpy
from onnx.backend.test import BackendTest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.onnxrt.backend_py import OnnxInferenceBackend
from mlprodict.onnx_conv import to_onnx
from mlprodict.npy.xop import loadop
from mlprodict.onnxrt import (
    backend_py, backend_ort, backend_micropy, backend_shape,
    backend_pyeval)


class TestCliBackend(ExtTestCase):

    def test_backend_class(self):
        backend_test = BackendTest(OnnxInferenceBackend, __name__)
        reg = re.compile("test_.*abs.*_cpu")
        cases = backend_test.test_cases
        test_cases = {}
        for _, v in cases.items():
            meths = []
            for meth in dir(v):
                if not reg.search(meth):
                    continue
                meths.append(getattr(v, meth))
            if len(meths) == 0:
                continue
            test_cases[v] = meths
        self.assertGreater(len(test_cases), 1)
        for te, meths in test_cases.items():
            inst = te()
            inst.setUp()
            for m in meths:
                with self.subTest(suite=te, meth=m):
                    m(inst)
                    pass

    def test_backend_iris_onnx(self):
        temp = get_temp_folder(__file__, 'temp_backend_iris_onnx')
        model_file = os.path.join(temp, "logreg_iris.onnx")
        data = load_iris()
        X, Y = data.data, data.target
        logreg = LogisticRegression(C=1e5).fit(X, Y)
        model = to_onnx(logreg, X.astype(numpy.float32),
                        options={'zipmap': False})
        with open(model_file, "wb") as f:
            f.write(model.SerializeToString())

        rep = backend_py.prepare(model_file, 'CPU')
        x = numpy.array([[-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0]],
                        dtype=numpy.float32)
        label, proba = rep.run(x)
        self.assertEqualArray(label, numpy.array([1, 1, 1]))
        self.assertEqual((3, 3), proba.shape)

    def test_backend_iris_onnx_ort(self):
        temp = get_temp_folder(__file__, 'temp_backend_iris_onnx')
        model_file = os.path.join(temp, "logreg_iris.onnx")
        data = load_iris()
        X, Y = data.data, data.target
        logreg = LogisticRegression(C=1e5).fit(X, Y)
        model = to_onnx(logreg, X.astype(numpy.float32),
                        options={'zipmap': False})
        with open(model_file, "wb") as f:
            f.write(model.SerializeToString())

        rep = backend_ort.prepare(model_file, 'CPU')
        x = numpy.array([[-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0]],
                        dtype=numpy.float32)
        label, proba = rep.run(x)
        self.assertEqualArray(label, numpy.array([1, 1, 1]))
        self.assertEqual((3, 3), proba.shape)

    def test_backend_onnx_micro(self):
        temp = get_temp_folder(__file__, 'temp_backend_micro')
        model_file = os.path.join(temp, "model.onnx")

        opset = 15
        dtype = numpy.float32
        OnnxAdd = loadop('Add')
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=opset)
        cop4 = OnnxAdd(cop, numpy.array([2], dtype=dtype), op_version=opset,
                       output_names=['Y'])
        model_def = cop4.to_onnx({'X': x}, target_opset=opset)
        with open(model_file, "wb") as f:
            f.write(model_def.SerializeToString())

        rep = backend_micropy.prepare(model_file, 'CPU')
        x = numpy.array([[-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0]],
                        dtype=numpy.float32)
        res = rep.run(x)[0]
        self.assertEqual((3, 4), res.shape)

    def test_backend_onnx_shape(self):
        temp = get_temp_folder(__file__, 'temp_backend_shape')
        model_file = os.path.join(temp, "model.onnx")

        opset = 15
        dtype = numpy.float32
        OnnxAdd = loadop('Add')
        x = numpy.array([1, 2, 4, 5, 5, 4, 1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 4))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=opset)
        cop4 = OnnxAdd(cop, numpy.array([2], dtype=dtype), op_version=opset,
                       output_names=['Y'])
        model_def = cop4.to_onnx({'X': x}, target_opset=opset)
        with open(model_file, "wb") as f:
            f.write(model_def.SerializeToString())

        rep = backend_shape.prepare(model_file, 'CPU')
        x = numpy.array([[-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0]],
                        dtype=numpy.float32)
        res = rep.run(x)[0]
        self.assertEqual((3, 4), tuple(res.shape))

    def test_backend_onnx_pyeval(self):
        temp = get_temp_folder(__file__, 'temp_backend_shape')
        model_file = os.path.join(temp, "model.onnx")

        opset = 15
        dtype = numpy.float32
        OnnxAdd = loadop('Add')
        x = numpy.array([1, 2, 4, 5, 5, 4, 1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 4))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=opset)
        cop4 = OnnxAdd(cop, numpy.array([2], dtype=dtype), op_version=opset,
                       output_names=['Y'])
        model_def = cop4.to_onnx({'X': x}, target_opset=opset)
        with open(model_file, "wb") as f:
            f.write(model_def.SerializeToString())

        rep = backend_pyeval.prepare(model_file, 'CPU')
        x = numpy.array([[-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0],
                         [-1.0, -2.0, -3.0, -4.0]],
                        dtype=numpy.float32)
        res = rep.run(x)[0]
        self.assertEqual((3, 4), tuple(res.shape))


if __name__ == "__main__":
    unittest.main()
