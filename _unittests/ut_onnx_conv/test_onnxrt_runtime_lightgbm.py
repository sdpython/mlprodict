"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import pandas
from lightgbm import LGBMClassifier
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
import skl2onnx
from skl2onnx.common.data_types import (
    StringTensorType, FloatTensorType, Int64TensorType,
    BooleanTensorType
)
from mlprodict.onnxrt import OnnxInference, to_onnx
from mlprodict.onnx_conv import register_converters


class TestOnnxrtRuntimeLightGbm(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxrt_python_lightgbm_categorical(self):

        X = pandas.DataFrame({"A": numpy.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
                              # int
                              "B": numpy.random.permutation([1, 2, 3] * 100),
                              # float
                              "C": numpy.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),
                              # bool
                              "D": numpy.random.permutation([True, False] * 150),
                              "E": pandas.Categorical(numpy.random.permutation(['z', 'y', 'x', 'w', 'v'] * 60),
                                                      ordered=True)})  # str and ordered categorical
        y = numpy.random.permutation([0, 1] * 150)
        X_test = pandas.DataFrame({"A": numpy.random.permutation(['a', 'b', 'e'] * 20),  # unseen category
                                   "B": numpy.random.permutation([1, 3] * 30),
                                   "C": numpy.random.permutation([0.1, -0.1, 0.2, 0.2] * 15),
                                   "D": numpy.random.permutation([True, False] * 30),
                                   "E": pandas.Categorical(numpy.random.permutation(['z', 'y'] * 30),
                                                           ordered=True)})
        cat_cols_actual = ["A", "B", "C", "D"]
        X[cat_cols_actual] = X[cat_cols_actual].astype('category')
        X_test[cat_cols_actual] = X_test[cat_cols_actual].astype('category')
        gbm0 = LGBMClassifier().fit(X, y)
        exp = gbm0.predict(X_test, raw_score=False)
        self.assertNotEmpty(exp)

        init_types = [('A', StringTensorType()),
                      ('B', Int64TensorType()),
                      ('C', FloatTensorType()),
                      ('D', BooleanTensorType()),
                      ('E', StringTensorType())]
        self.assertRaise(lambda: to_onnx(gbm0, initial_types=init_types), RuntimeError,
                         "at most 1 input(s) is(are) supported")

        X = X[['C']].values.astype(numpy.float32)
        X_test = X_test[['C']].values.astype(numpy.float32)
        gbm0 = LGBMClassifier().fit(X, y, categorical_feature=[0])
        exp = gbm0.predict_proba(X_test, raw_score=False)
        model_def = to_onnx(gbm0, X)

        oinf = OnnxInference(model_def)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), ['label', 'probabilities'])
        df = pandas.DataFrame(y['probabilities'])
        self.assertEqual(df.shape, (X_test.shape[0], 2))
        self.assertEqual(exp.shape, (X_test.shape[0], 2))
        # self.assertEqualArray(exp, df.values, decimal=6)


if __name__ == "__main__":
    unittest.main()
