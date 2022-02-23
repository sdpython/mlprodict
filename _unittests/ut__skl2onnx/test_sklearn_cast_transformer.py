"""
@brief      test tree node (time=15s)
"""
import unittest
import math
import numpy
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from skl2onnx.sklapi import CastTransformer
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    Int64TensorType, FloatTensorType, DoubleTensorType)
from mlprodict.testing.test_utils import dump_data_and_model
from mlprodict.tools.ort_wrapper import InferenceSession
from mlprodict import __max_supported_opset__ as TARGET_OPSET


class TestSklearnCastTransformerConverter(unittest.TestCase):

    def common_test_cast_transformer(self, dtype, input_type):
        model = Pipeline([
            ('cast', CastTransformer(dtype=dtype)),
            ('invcast', CastTransformer(dtype=numpy.float32)),
        ])
        data = numpy.array([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]],
                           dtype=numpy.float32)
        model.fit(data)
        pred = model.steps[0][1].transform(data)
        self.assertEqual(pred.dtype, dtype)
        model_onnx = convert_sklearn(
            model, "cast", [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnCastTransformer{}".format(
                input_type.__class__.__name__))

    def test_cast_transformer_float(self):
        self.common_test_cast_transformer(
            numpy.float32, FloatTensorType)

    def test_cast_transformer_float64(self):
        self.common_test_cast_transformer(
            numpy.float64, DoubleTensorType)

    def test_cast_transformer_int64(self):
        self.common_test_cast_transformer(
            numpy.int64, Int64TensorType)

    def test_pipeline(self):

        def maxdiff(a1, a2):
            d = numpy.abs(a1.ravel() - a2.ravel())
            return d.max()

        X, y = make_regression(  # pylint: disable=W0632
            10000, 10, random_state=3)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=3)
        Xi_train, yi_train = X_train.copy(), y_train.copy()
        Xi_test = X_test.copy()
        for i in range(X.shape[1]):
            Xi_train[:, i] = (Xi_train[:, i] * math.pi * 2 ** i).astype(
                numpy.int64)
            Xi_test[:, i] = (Xi_test[:, i] * math.pi * 2 ** i).astype(
                numpy.int64)
        max_depth = 10
        Xi_test = Xi_test.astype(numpy.float32)

        # model 1
        model1 = Pipeline([
            ('scaler', StandardScaler()),
            ('dt', DecisionTreeRegressor(max_depth=max_depth))])
        model1.fit(Xi_train, yi_train)
        exp1 = model1.predict(Xi_test)
        onx1 = to_onnx(model1, X_train[:1].astype(numpy.float32),
                       target_opset=TARGET_OPSET)
        sess1 = InferenceSession(onx1.SerializeToString())
        got1 = sess1.run(None, {'X': Xi_test})[0]
        md1 = maxdiff(exp1, got1)

        # model 2
        model2 = Pipeline([
            ('cast64', CastTransformer(dtype=numpy.float64)),
            ('scaler', StandardScaler()),
            ('cast', CastTransformer()),
            ('dt', DecisionTreeRegressor(max_depth=max_depth))])
        model2.fit(Xi_train, yi_train)
        exp2 = model2.predict(Xi_test)
        onx = to_onnx(model2, X_train[:1].astype(numpy.float32),
                      options={StandardScaler: {'div': 'div_cast'}},
                      target_opset=TARGET_OPSET)
        sess2 = InferenceSession(onx.SerializeToString())
        got2 = sess2.run(None, {'X': Xi_test})[0]
        md2 = maxdiff(exp2, got2)
        self.assertLess(md2, md1)
        self.assertLess(md2, 0.01)


if __name__ == "__main__":
    unittest.main()
