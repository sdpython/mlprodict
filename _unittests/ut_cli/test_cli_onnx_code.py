"""
@brief      test tree node (time=15s)
"""
import os
import unittest
import numpy
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from mlprodict.__main__ import main
from mlprodict import __max_supported_opsets__
from mlprodict.onnx_conv import to_onnx


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

    def test_cli_plot_onnx(self):
        temp = get_temp_folder(__file__, "temp_cli_plot_onnx")
        name = os.path.join(
            temp, "..", "..", "ut_tools", "data", "fft2d_any.onnx")
        self.assertExists(name)
        for fmt in ['simple', 'dot', 'io', 'raw']:
            with self.subTest(fmt=fmt):
                output = os.path.join(temp, "code_%s.py" % fmt)
                st = BufferedPrint()
                main(args=["plot_onnx", "--filename", name, '--format', fmt,
                           "--output", output, "--verbose", "1"], fLOG=st.fprint)
                self.assertExists(output)

    def test_cli_plot_onnx_tree(self):
        temp = get_temp_folder(__file__, "temp_cli_plot_onnx_tree")

        X, y = make_regression(n_features=2)  # pylint: disable=W0632
        tree = DecisionTreeRegressor()
        tree.fit(X, y)
        onx = to_onnx(tree, X.astype(numpy.float32),
                      target_opset=__max_supported_opsets__)
        name = os.path.join(temp, "tree.onnx")
        with open(name, "wb") as f:
            f.write(onx.SerializeToString())
        self.assertExists(name)
        for fmt in ['tree', 'mat']:
            with self.subTest(fmt=fmt):
                output = os.path.join(temp, "code_%s.py" % fmt)
                st = BufferedPrint()
                main(args=["plot_onnx", "--filename", name, '--format', fmt,
                           "--output", output, "--verbose", "1"], fLOG=st.fprint)
                self.assertExists(output)


if __name__ == "__main__":
    unittest.main()
