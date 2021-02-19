"""
@brief      test tree node (time=14s)
"""
import unittest
import numpy
from sklearn.ensemble import IsolationForest
from skl2onnx import to_onnx
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnIsolationForest(ExtTestCase):

    def test_isolation_forest(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = numpy.array([[-1.1, -1.2], [0.3, 0.2],
                            [0.5, 0.4], [100., 99.]], dtype=numpy.float32)
        model = isol.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            data, model, model_onnx, basename="IsolationForest",
            backend=('python', ), methods=['predict', 'decision_function'])

    def test_isolation_forest_rnd(self):
        isol = IsolationForest(n_estimators=2, random_state=0)
        rs = numpy.random.RandomState(0)  # pylint: disable=E1101
        data = rs.randn(100, 4).astype(numpy.float32)
        data[-1, 2:] = 99.
        data[-2, :2] = -99.
        model = isol.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            data[5:10], model, model_onnx, basename="IsolationForestRnd",
            backend=('python', ), methods=['predict', 'decision_function'],
            verbose=False)
        dump_data_and_model(
            data, model, model_onnx, basename="IsolationForestRnd",
            backend=('python', ), methods=['predict', 'decision_function'],
            verbose=False)

    def test_isolation_forest_op1(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = numpy.array([[-1.1, -1.2], [0.3, 0.2],
                            [0.5, 0.4], [100., 99.]], dtype=numpy.float32)
        model = isol.fit(data)
        with self.assertRaises(RuntimeError):
            to_onnx(model, data,
                    target_opset={'': TARGET_OPSET, 'ai.onnx.ml': 1})


if __name__ == '__main__':
    unittest.main()
