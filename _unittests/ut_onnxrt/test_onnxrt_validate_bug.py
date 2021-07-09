"""
@brief      test log(time=40s)
"""
import os
import unittest
import numpy
import onnx
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxMatMul  # pylint: disable=E0611
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools import get_opset_number_from_onnx
from mlprodict.tools.ort_wrapper import InferenceSession


class TestOnnxrtValidateBug(ExtTestCase):

    def test_bug_add(self):
        coef = numpy.array([-8.43436238e-02, 5.47765517e-02, 6.77578341e-02, 1.56675273e+00,
                            -1.45737317e+01, 3.78662574e+00 - 6.52943746e-03 - 1.39463522e+00,
                            2.89157796e-01 - 1.53753213e-02 - 9.88045749e-01, 1.00224585e-02,
                            -4.96820220e-01], dtype=numpy.float64)
        intercept = 35.672858515632

        X_test = (coef + 1.).reshape((1, coef.shape[0]))

        onnx_fct = OnnxAdd(
            OnnxMatMul('X', coef.astype(numpy.float64),
                       op_version=get_opset_number_from_onnx()),
            numpy.array([intercept]), output_names=['Y'],
            op_version=get_opset_number_from_onnx())
        onnx_model64 = onnx_fct.to_onnx({'X': X_test.astype(numpy.float64)})

        oinf = OnnxInference(onnx_model64)
        ort_pred = oinf.run({'X': X_test.astype(numpy.float64)})['Y']
        self.assertEqualArray(ort_pred, numpy.array([245.19907295849504]))

    def test_dict_vectorizer_rfr(self):
        this = os.path.abspath(os.path.dirname(__file__))
        data = os.path.join(this, "data", "pipeline_vectorize.onnx")
        sess = InferenceSession(data)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "float_input")
        input_type = str(sess.get_inputs()[0].type)
        self.assertEqual(input_type, "map(int64,tensor(float))")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "variable1")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(float)")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [1, 1])

        x = {0: 25.0, 1: 5.13, 2: 0.0, 3: 0.453, 4: 5.966}
        res = sess.run([output_name], {input_name: x})

        model_onnx = onnx.load(data)
        oinf = OnnxInference(model_onnx, runtime='onnxruntime1')
        res2 = oinf.run({input_name: x})

        x = {k: numpy.float32(v) for k, v in x.items()}
        oinf = OnnxInference(model_onnx, runtime='python')
        # , verbose=10, fLOG=print)
        res3 = oinf.run({input_name: numpy.array([x])})

        self.assertEqualFloat(res[0][0, 0], res2["variable1"][0, 0])
        self.assertEqualFloat(res[0][0, 0], res3["variable1"][0])


if __name__ == "__main__":
    unittest.main()
