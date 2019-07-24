"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from scipy.spatial.distance import squareform, pdist, cdist as scipy_cdist
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxAdd
)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntimeScan(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_pdist(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('input', 'input')
        cdist = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())])

        sess = OnnxInference(model_def)
        res = sess.run({'input': x})
        self.assertEqual(list(res.keys()), ['cdist'])

        exp = squareform(pdist(x * 2, metric="sqeuclidean"))
        self.assertEqualArray(exp, res['cdist'])

        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((2, 3))
        res = sess.run({'input': x})
        self.assertEqual(list(res.keys()), ['cdist'])

    def test_onnx_example_cdist_in(self):
        from skl2onnx.algebra.complex_functions import onnx_cdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        x2 = numpy.array([1.1, 2.1, 4.01, 5.01, 5.001, 4.001, 0, 0]).astype(
            numpy.float32).reshape((4, 2))
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(onnx_cdist(cop, x2, dtype=numpy.float32),
                            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType(None, None))])

        sess = OnnxInference(model_def)
        res = sess.run({'input': x})
        exp = scipy_cdist(x * 2, x2, metric="sqeuclidean")
        self.assertEqualArray(exp, res['cdist'], decimal=5)

        x = numpy.array([[6.1, 2.8, 4.7, 1.2],
                         [5.7, 3.8, 1.7, 0.3],
                         [7.7, 2.6, 6.9, 2.3],
                         [6., 2.9, 4.5, 1.5],
                         [6.8, 2.8, 4.8, 1.4],
                         [5.4, 3.4, 1.5, 0.4],
                         [5.6, 2.9, 3.6, 1.3],
                         [6.9, 3.1, 5.1, 2.3]], dtype=numpy.float32)
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(onnx_cdist(cop, x, dtype=numpy.float32),
                            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())])

        sess = OnnxInference(model_def)
        res = sess.run({'input': x})
        exp = scipy_cdist(x * 2, x, metric="sqeuclidean")
        self.assertEqualArray(exp, res['cdist'], decimal=4)

    def test_onnx_example_cdist_bigger(self):

        from skl2onnx.algebra.complex_functions import onnx_cdist
        data = load_iris()
        X, y = data.data, data.target
        self.assertNotEmpty(y)
        X_train = X[::2]
        # y_train = y[::2]
        X_test = X[1::2]
        # y_test = y[1::2]
        onx = OnnxIdentity(onnx_cdist(OnnxIdentity('X'), X_train.astype(numpy.float32),
                                      metric="euclidean", dtype=numpy.float32),
                           output_names=['Y'])
        final = onx.to_onnx(inputs=[('X', FloatTensorType([None, None]))],
                            outputs=[('Y', FloatTensorType())])

        oinf = OnnxInference(final, runtime="python")
        res = oinf.run({'X': X_train.astype(numpy.float32)})['Y']
        exp = scipy_cdist(X_train, X_train, metric="euclidean")
        self.assertEqualArray(exp, res, decimal=6)
        res = oinf.run({'X': X_test.astype(numpy.float32)})['Y']
        exp = scipy_cdist(X_test, X_train, metric="euclidean")
        self.assertEqualArray(exp, res, decimal=6)


if __name__ == "__main__":
    unittest.main()
