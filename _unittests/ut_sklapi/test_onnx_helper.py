"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.helpers.onnx_helper import (
    load_onnx_model, save_onnx_model, select_model_inputs_outputs,
    enumerate_model_node_outputs)
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools.asv_options_helper import get_ir_version_from_onnx
from mlprodict.tools.ort_wrapper import InferenceSession


class TestOnnxHelper(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def get_model(self, model):
        session = InferenceSession(save_onnx_model(model))
        return lambda X: session.run(None, {'input': X})[0]

    def test_onnx_helper_load_save(self):
        model = make_pipeline(StandardScaler(), Binarizer(threshold=0.5))
        X = numpy.array([[0.1, 1.1], [0.2, 2.2]])
        model.fit(X)
        model_onnx = convert_sklearn(
            model, 'binarizer', [('input', FloatTensorType([None, 2]))])
        model_onnx.ir_version = get_ir_version_from_onnx()
        filename = "temp_onnx_helper_load_save.onnx"
        save_onnx_model(model_onnx, filename)
        model = load_onnx_model(filename)
        list(enumerate_model_node_outputs(model))
        new_model = select_model_inputs_outputs(model, 'variable')
        self.assertTrue(new_model.graph is not None)  # pylint: disable=E1101

        tr1 = self.get_model(model)
        tr2 = self.get_model(new_model)
        X = X.astype(numpy.float32)
        X1 = tr1(X)
        X2 = tr2(X)
        self.assertEqual(X1.shape, (2, 2))
        self.assertEqual(X2.shape, (2, 2))

    def test_onnx_helper_load_save_init(self):
        model = make_pipeline(
            Binarizer(),
            OneHotEncoder(sparse=False, handle_unknown='ignore'),
            StandardScaler())
        X = numpy.array([[0.1, 1.1], [0.2, 2.2], [0.4, 2.2], [0.2, 2.4]])
        model.fit(X)
        model_onnx = convert_sklearn(
            model, 'pipe3', [('input', FloatTensorType([None, 2]))])
        model_onnx.ir_version = get_ir_version_from_onnx()
        filename = "temp_onnx_helper_load_save.onnx"
        save_onnx_model(model_onnx, filename)
        model = load_onnx_model(filename)
        list(enumerate_model_node_outputs(model))
        new_model = select_model_inputs_outputs(model, 'variable')
        self.assertTrue(new_model.graph is not None)  # pylint: disable=E1101
        tr1 = self.get_model(model)
        tr2 = self.get_model(new_model)
        X = X.astype(numpy.float32)
        X1 = tr1(X)
        X2 = tr2(X)
        self.assertEqual(X1.shape, (4, 2))
        self.assertEqual(X2.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
