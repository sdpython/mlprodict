"""
@brief      test log(time=5s)
"""
import unittest
from logging import getLogger
import numpy
from pandas import DataFrame
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.algebra.onnx_ops import OnnxAdd  # pylint: disable=E0611
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
from mlprodict.sklapi import OnnxTransformer
from mlprodict.onnxrt import OnnxInference


class TestInferenceSessionOnnx2Onnx(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_pipeline(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        pca = PCA(n_components=2)
        pca.fit(X)

        onx = convert_sklearn(pca, initial_types=[
                              ('input', FloatTensorType((None, X.shape[1])))])
        onx_bytes = onx.SerializeToString()
        tr = OnnxTransformer(onx_bytes)

        pipe = make_pipeline(tr, LogisticRegression())
        pipe.fit(X, y)
        pred = pipe.predict(X)
        self.assertEqual(pred.shape, (150, ))
        skl_pred = pca.transform(X)
        skl_onx = pipe.steps[0][1].transform(X)
        self.assertEqualArray(skl_pred, skl_onx, decimal=5)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_pipeline_add(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        pca = PCA(n_components=2)
        pca.fit(X)

        add = OnnxAdd('X', numpy.full((1, X.shape[1]), 1, dtype=numpy.float32),
                      output_names=['Yadd'])
        onx = add.to_onnx(inputs=[('X', FloatTensorType((None, X.shape[1])))],
                          outputs=[('Yadd', FloatTensorType((None, X.shape[1])))])

        tr = OnnxTransformer(onx)
        tr.fit()

        pipe = make_pipeline(tr, LogisticRegression())
        pipe.fit(X, y)
        pred = pipe.predict(X)
        self.assertEqual(pred.shape, (150, ))
        model_onnx = to_onnx(pipe, X.astype(numpy.float32))

        oinf = OnnxInference(model_onnx)
        y1 = pipe.predict(X)
        y2 = oinf.run({'X': X.astype(numpy.float32)})
        self.assertEqual(list(y2), ['output_label', 'output_probability'])
        self.assertEqualArray(y1, y2['output_label'])
        y1 = pipe.predict_proba(X)
        probas = DataFrame(list(y2['output_probability'])).values
        self.assertEqualArray(y1, probas, decimal=5)


if __name__ == '__main__':
    unittest.main()
