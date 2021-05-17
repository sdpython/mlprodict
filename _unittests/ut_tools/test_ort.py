"""
@brief      test log(time=3s)
"""
import unittest
import os
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxMul, OnnxSub)
from mlprodict.tools.ort_wrapper import prepare_c_profiling


class TestOrt(ExtTestCase):

    def test_prepare_c_profiling(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=13)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=13)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype), op_version=13,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=13), cop2, output_names=['final'],
            op_version=13)
        model_def = cop4.to_onnx({'X': x})

        temp = get_temp_folder(__file__, "temp_prepare_c_profiling")
        cmd = prepare_c_profiling(model_def, [x], dest=temp)
        self.assertStartsWith("onnxruntime_perf_test", cmd)
        self.assertExists(os.path.join(temp, "test_data_set_0", "input_0"))
        self.assertExists(os.path.join(temp, "test_data_set_0", "output_0"))


if __name__ == "__main__":
    unittest.main()
