"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxMul, OnnxAdd)
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.asv_options_helper import (
    get_ir_version_from_onnx, get_opset_number_from_onnx)


class TestOnnxrtOnnxRuntimeRuntime(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_runtime_add(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)

        model_def.ir_version = get_ir_version_from_onnx()
        oinf = OnnxInference(model_def, runtime='onnxruntime1')
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(idi + X, got['Y'], decimal=6)

        oinf = OnnxInference(model_def, runtime='onnxruntime2')
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(idi + X, got['Y'], decimal=6)

        oinf = OnnxInference(model_def, runtime='onnxruntime1')
        got = oinf.run({'X': X}, intermediate=True)
        self.assertEqual(list(sorted(got)), ['Ad_Addcst', 'X', 'Y'])
        self.assertEqualArray(idi + X, got['Y'], decimal=6)

    def test_onnxt_runtime_add_raise(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        self.assertRaise(lambda: OnnxInference(model_def, runtime='onnxruntime-1'),
                         ValueError)

    def test_onnxt_runtime_add1(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
        model_def.ir_version = get_ir_version_from_onnx()
        oinf = OnnxInference(model_def, runtime='onnxruntime1')
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(idi + X, got['Y'], decimal=6)

    def test_onnxruntime_bug(self):
        rnd = numpy.random.randn(2, 20, 20).astype(numpy.float32)
        bin = (numpy.random.random((20, 20)).astype(
            numpy.float32) >= 0.7).astype(numpy.float32)
        mul = rnd * bin
        isn = any(numpy.isnan(mul.ravel()))
        self.assertFalse(isn)

        node = OnnxMul('X', bin, output_names=['Y'],
                       op_version=get_opset_number_from_onnx())
        onx = node.to_onnx({'X': rnd})
        for rt in ['python', 'onnxruntime1']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(onx, runtime=rt)
                y = oinf.run({'X': rnd})['Y']
                self.assertEqualArray(mul, y)


if __name__ == "__main__":
    unittest.main()
