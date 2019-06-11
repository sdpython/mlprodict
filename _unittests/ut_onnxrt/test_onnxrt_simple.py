"""
@brief      test log(time=2s)
"""
import os
from io import BytesIO
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from skl2onnx.algebra.onnx_ops import OnnxAdd  # pylint: disable=E0611
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtSimple(ExtTestCase):

    def test_onnxt_idi(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})

        oinf = OnnxInference(model_def)
        res = str(oinf)
        self.assertIn('op_type: "Add"', res)

        sb = model_def.SerializeToString()
        oinf = OnnxInference(sb)
        res = str(oinf)
        self.assertIn('op_type: "Add"', res)

        sb = BytesIO(model_def.SerializeToString())
        oinf = OnnxInference(sb)
        res = str(oinf)
        self.assertIn('op_type: "Add"', res)

        temp = get_temp_folder(__file__, "temp_onnxrt_idi")
        name = os.path.join(temp, "m.onnx")
        with open(name, "wb") as f:
            f.write(model_def.SerializeToString())

        oinf = OnnxInference(name)
        res = str(oinf)
        self.assertIn('op_type: "Add"', res)


if __name__ == "__main__":
    unittest.main()
