"""
@brief      test tree node (time=10s)
"""
import os
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.__main__ import main


class TestCliOnnxCode(ExtTestCase):

    def test_cli_onnx_code(self):
        st = BufferedPrint()
        main(args=["onnx_code", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("verbose", res)

    def test_cli_onnx_code_onnx(self):
        temp = get_temp_folder(__file__, "temp_cli_onnx_code_onnx")
        name = os.path.join(
            temp, "..", "..", "ut_tools", "data", "fft2d_any.onnx")
        self.assertExists(name)
        output = os.path.join(temp, "code_onnx.py")
        st = BufferedPrint()
        main(args=["onnx_code", "--filename", name,
                   "--output", output, "--verbose", "1"], fLOG=st.fprint)
        self.assertExists(output)
        with open(output, "r", encoding='utf-8') as f:
            content = f.read()
        self.assertIn("create_model()", content)

    def test_cli_onnx_code_tf2onnx(self):
        temp = get_temp_folder(__file__, "temp_cli_onnx_code_tf2onnx")
        name = os.path.join(
            temp, "..", "..", "ut_tools", "data", "fft2d_any.onnx")
        self.assertExists(name)
        output = os.path.join(temp, "code_tf2onnx.py")
        st = BufferedPrint()
        main(args=["onnx_code", "--filename", name, '--format', 'tf2onnx',
                   "--output", output, "--verbose", "1"], fLOG=st.fprint)
        self.assertExists(output)
        with open(output, "r", encoding='utf-8') as f:
            content = f.read()
        self.assertIn("tf_op", content)

    def test_cli_onnx_code_numpy(self):
        temp = get_temp_folder(__file__, "temp_cli_onnx_code_numpy")
        name = os.path.join(
            temp, "..", "..", "ut_tools", "data", "fft2d_any.onnx")
        self.assertExists(name)
        output = os.path.join(temp, "code_numpy.py")
        st = BufferedPrint()
        main(args=["onnx_code", "--filename", name, '--format', 'numpy',
                   "--output", output, "--verbose", "1"], fLOG=st.fprint)
        self.assertExists(output)
        with open(output, "r", encoding='utf-8') as f:
            content = f.read()
        self.assertIn("def numpy_", content)


if __name__ == "__main__":
    unittest.main()
