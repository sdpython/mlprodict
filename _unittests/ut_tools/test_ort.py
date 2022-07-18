"""
@brief      test log(time=6s)
"""
import unittest
import os
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.npy.xop import loadop
from mlprodict.tools.ort_wrapper import prepare_c_profiling
from mlprodict.tools.onnx_inference_ort_helper import (
    get_ort_device, device_to_providers)


class TestOrt(ExtTestCase):

    opset = 15  # opset = 13, 14, ...

    def test_prepare_c_profiling(self):
        OnnxAdd, OnnxMul, OnnxSub = loadop('Add', 'Mul', 'Sub')
        opset = TestOrt.opset
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=opset)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=opset)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype), op_version=opset,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=opset), cop2, output_names=['final'],
            op_version=13)
        model_def = cop4.to_onnx({'X': x}, target_opset=opset)

        temp = get_temp_folder(__file__, "temp_prepare_c_profiling")
        cmd = prepare_c_profiling(model_def, [x], dest=temp)
        self.assertStartsWith("onnx", cmd)
        self.assertExists(os.path.join(temp, "model.onnx"))
        self.assertExists(os.path.join(temp, "test_data_set_0", "input_0.pb"))
        self.assertExists(os.path.join(temp, "test_data_set_0", "output_0.pb"))

    def test_get_ort_device(self):
        self.assertEqual(get_ort_device('gpu').device_type(), 1)
        self.assertEqual(get_ort_device('cuda:0').device_type(), 1)
        self.assertEqual(get_ort_device('cuda').device_type(), 1)
        self.assertEqual(get_ort_device('gpu:0').device_type(), 1)
        self.assertEqual(get_ort_device('gpu:0').device_type(), 1)

    def test_device_to_providers(self):
        self.assertEqual(device_to_providers('cpu'), ['CPUExecutionProvider'])
        self.assertEqual(device_to_providers('cuda'),
                         ['CUDAExecutionProvider', 'CPUExecutionProvider'])


if __name__ == "__main__":
    unittest.main()
