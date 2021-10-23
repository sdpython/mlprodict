"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import (
    ExtTestCase, skipif_appveyor, skipif_circleci,
    skipif_travis, skipif_azure)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestBugsOnnxrtOnnxConverter(ExtTestCase):

    @skipif_appveyor("old version of onnxconvert-common")
    @skipif_circleci("old version of onnxconvert-common")
    @skipif_travis("old version of onnxconvert-common")
    @skipif_azure("old version of onnxconvert-common")
    def test_bug_apply_clip(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        y_train = y_train.astype(numpy.float32)
        clr = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=3)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                            target_opset=12)

        oinf2 = OnnxInference(model_def, runtime='python_compiled')
        res = oinf2.run({'X': X_test[:5]})
        self.assertGreater(len(res), 1)


if __name__ == "__main__":
    unittest.main()
