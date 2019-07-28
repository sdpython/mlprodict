"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import to_onnx
from mlprodict.onnxrt.model_checker import onnx_shaker
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtModelChecker(ExtTestCase):

    def test_onnxt_model_checker(self):
        arr = numpy.array([1.111111111111111,
                           -1.11111111111111111,
                           2.222222222222222222,
                           -2.22222222222222222])
        conv = arr.astype(numpy.float32)
        delta = numpy.abs(arr - conv)
        pos = numpy.sum(delta > 0)
        self.assertEqual(pos, 4)

    def test_onnx_shaker(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=1, shuffle=True)
        clr = GradientBoostingClassifier(n_estimators=20)
        clr.fit(X_train, y_train)
        exp = clr.predict_proba(X_test)[:, 2]

        def output_fct(res):
            val = res['output_probability'].values
            return val[:, 2]

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        inputs = {'X': X_test}
        res1 = output_fct(oinf.run({'X': X_test.astype(numpy.float32)}))
        shaked = onnx_shaker(oinf, inputs, dtype=numpy.float32, n=100,
                             output_fct=output_fct, force=2)
        delta1 = numpy.max(shaked.max(axis=1) - shaked.min(axis=1))
        deltae = numpy.max(numpy.abs(res1 - exp))
        self.assertLesser(deltae, delta1 * 2)


if __name__ == "__main__":
    unittest.main()
