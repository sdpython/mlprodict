"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import OnnxAdd  # pylint: disable=E0611
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtOnnxRuntimeRuntimeWhole(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_runtime_add_raise(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        self.assertRaise(lambda: OnnxInference(model_def, runtime='onnxruntime_whole'),
                         ValueError)

    def test_onnxt_runtime_add(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
        oinf = OnnxInference(model_def, runtime='onnxruntime-whole')
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(idi + X, got['Y'], decimal=6)


if __name__ == "__main__":
    unittest.main()
