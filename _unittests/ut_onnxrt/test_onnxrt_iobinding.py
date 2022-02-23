"""
@brief      test log(time=6s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611,W0611
    OrtDevice as C_OrtDevice, OrtValue as C_OrtValue)
from onnxruntime import get_device
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611,W0611
    OnnxAdd)
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.onnx_inference_ort_helper import get_ort_device
from mlprodict import __max_supported_opset__ as TARGET_OPSET


DEVICE = "cuda" if get_device().upper() == 'GPU' else 'cpu'


class TestOnnxrtIOBinding(ExtTestCase):

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_cpu_numpy_python(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        idi2 = (numpy.identity(2) * 2).astype(numpy.float32)
        onx = OnnxAdd(
            OnnxAdd('X', idi, op_version=TARGET_OPSET),
            idi2, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        X = numpy.array([[1, 1], [3, 3]])
        y = oinf.run({'X': X.astype(numpy.float32)})
        exp = numpy.array([[4, 1], [3, 6]], dtype=numpy.float32)
        self.assertEqual(list(y), ['Y'])
        self.assertEqualArray(y['Y'], exp)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_cpu_numpy_onnxruntime1(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        idi2 = (numpy.identity(2) * 2).astype(numpy.float32)
        onx = OnnxAdd(
            OnnxAdd('X', idi, op_version=TARGET_OPSET),
            idi2, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime="onnxruntime1")
        X = numpy.array([[1, 1], [3, 3]])
        y = oinf.run({'X': X.astype(numpy.float32)})
        exp = numpy.array([[4, 1], [3, 6]], dtype=numpy.float32)
        self.assertEqual(list(y), ['Y'])
        self.assertEqualArray(y['Y'], exp)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_cpu_ortvalue_python(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        idi2 = (numpy.identity(2) * 2).astype(numpy.float32)
        onx = OnnxAdd(
            OnnxAdd('X', idi, op_version=TARGET_OPSET),
            idi2, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def)
        X = numpy.array([[1, 1], [3, 3]])
        X32 = X.astype(numpy.float32)
        ov = C_OrtValue.ortvalue_from_numpy(X32, get_ort_device('cpu'))
        self.assertRaise(lambda: oinf.run({'X': ov}), AttributeError)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_cpu_ortvalue_ort(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        idi2 = (numpy.identity(2) * 2).astype(numpy.float32)
        onx = OnnxAdd(
            OnnxAdd('X', idi, op_version=TARGET_OPSET),
            idi2, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime="onnxruntime1")
        X = numpy.array([[1, 1], [3, 3]])
        X32 = X.astype(numpy.float32)
        ov = C_OrtValue.ortvalue_from_numpy(X32, get_ort_device('cpu'))
        y = oinf.run({'X': ov})
        exp = numpy.array([[4, 1], [3, 6]], dtype=numpy.float32)
        self.assertEqual(list(y), ['Y'])
        self.assertEqualArray(y['Y'].numpy(), exp)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_cpu_ortvalue_ort_cpu(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        idi2 = (numpy.identity(2) * 2).astype(numpy.float32)
        onx = OnnxAdd(
            OnnxAdd('X', idi, op_version=TARGET_OPSET),
            idi2, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        self.assertRaise(lambda: OnnxInference(model_def, device='cpu'),
                         ValueError)
        oinf = OnnxInference(model_def, runtime="onnxruntime1", device='cpu')
        X = numpy.array([[1, 1], [3, 3]])
        X32 = X.astype(numpy.float32)
        ov = C_OrtValue.ortvalue_from_numpy(X32, get_ort_device('cpu'))
        y = oinf.run({'X': ov})
        exp = numpy.array([[4, 1], [3, 6]], dtype=numpy.float32)
        self.assertEqual(list(y), ['Y'])
        self.assertEqualArray(y['Y'].numpy(), exp)

    @unittest.skipIf(DEVICE != 'cuda', reason="runs only on GPU")
    @ignore_warnings(DeprecationWarning)
    def test_onnxt_ortvalue_ort_gpu(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        idi2 = (numpy.identity(2) * 2).astype(numpy.float32)
        onx = OnnxAdd(
            OnnxAdd('X', idi, op_version=TARGET_OPSET),
            idi2, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime="onnxruntime1", device='cuda')
        X = numpy.array([[1, 1], [3, 3]])
        X32 = X.astype(numpy.float32)
        ov = C_OrtValue.ortvalue_from_numpy(X32, get_ort_device('cuda'))
        y = oinf.run({'X': ov})
        exp = numpy.array([[4, 1], [3, 6]], dtype=numpy.float32)
        self.assertEqual(list(y), ['Y'])
        self.assertEqualArray(y['Y'].cpu().numpy(), exp)


if __name__ == "__main__":
    unittest.main()
