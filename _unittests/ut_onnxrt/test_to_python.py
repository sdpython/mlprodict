"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import OnnxAdd  # pylint: disable=E0611
from mlprodict.onnxrt import OnnxInference


class TestToPython(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_runtime_add_except(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def, runtime='onnxruntime1')
        try:
            oinf.to_python()
        except ValueError:
            pass

    def test_onnxt_runtime_add(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        # X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
        oinf = OnnxInference(model_def, runtime='python')
        res = oinf.to_python()
        self.assertNotEmpty(res)
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 2)
        self.assertIn('onnx_pyrt_Ad_Addcst.pkl', res)
        self.assertIn('onnx_pyrt_main.py', res)
        cd = res['onnx_pyrt_main.py']
        self.assertIn('def pyrt_Add(X, Ad_Addcst):', cd)
        self.assertIn('def run(self, X):', cd)
        # print(cd)


if __name__ == "__main__":
    unittest.main()
