"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.custom_ops import (  # pylint: disable=E0611
    OnnxCDist, OnnxSolve
)
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx
from mlprodict.onnxrt.validate.validate_python import validate_python_inference


python_tested = []


class TestOnnxrtPythonRuntimeCustom(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        if __name__ == "__main__":
            import pprint
            print('\n-----------')
            pprint.pprint(
                list(sorted({_.__name__ for _ in python_tested})))
            print('-----------')

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_runtime_cdist(self):
        for metric in ['sqeuclidean', 'euclidean']:
            with self.subTest(metric=metric):
                X = numpy.array([[2, 1], [0, 1]], dtype=float)
                Y = numpy.array([[2, 1, 5], [0, 1, 3]], dtype=float).T
                Z = cdist(X, Y, metric=metric)

                onx = OnnxCDist('X', 'Y', output_names=['Z'],
                                metric=metric,
                                op_version=get_opset_number_from_onnx())
                model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                         'Y': Y.astype(numpy.float32)},
                                        outputs={'Z': Z.astype(numpy.float32)},
                                        target_opset=get_opset_number_from_onnx())
                self.assertIn('s: "%s"' % metric, str(model_def))
                oinf = OnnxInference(model_def)
                got = oinf.run({'X': X, 'Y': Y})
                self.assertEqual(list(sorted(got)), ['Z'])
                self.assertEqualArray(Z, got['Z'], decimal=6)

                python_tested.append(OnnxCDist)
                oinfpy = OnnxInference(
                    model_def, runtime="python", inplace=True)
                validate_python_inference(
                    oinfpy, {'X': X.astype(numpy.float32),
                             'Y': Y.astype(numpy.float32)},
                    tolerance=1e-6)

    def test_onnxt_runtime_solve(self):
        for transposed in [False, True]:
            with self.subTest(transposed=transposed):
                A = numpy.array([[2, 1], [0, 1]], dtype=float)
                Y = numpy.array([2, 1], dtype=float)
                X = solve(A, Y, transposed=transposed)

                onx = OnnxSolve('A', 'Y', output_names=['X'],
                                transposed=transposed,
                                op_version=get_opset_number_from_onnx())
                model_def = onx.to_onnx({'A': A.astype(numpy.float32),
                                         'Y': Y.astype(numpy.float32)},
                                        outputs={'X': X.astype(numpy.float32)},
                                        target_opset=get_opset_number_from_onnx())
                oinf = OnnxInference(model_def)
                got = oinf.run({'A': A, 'Y': Y})
                self.assertEqual(list(sorted(got)), ['X'])
                self.assertEqualArray(X, got['X'], decimal=6)

                python_tested.append(OnnxCDist)
                oinfpy = OnnxInference(
                    model_def, runtime="python", inplace=True)
                validate_python_inference(
                    oinfpy, {'A': A.astype(numpy.float32),
                             'Y': Y.astype(numpy.float32)})


if __name__ == "__main__":
    unittest.main()
