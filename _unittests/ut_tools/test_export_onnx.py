"""
@brief      test log(time=3s)
"""
import os
import unittest
import collections
import inspect
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import numpy
from onnx import numpy_helper, helper
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor, make_graph,
    make_tensor_value_info)
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_tools.onnx_export import export2onnx, export2tf2onnx
from mlprodict.testing.verify_code import verify_code
from mlprodict.onnxrt import OnnxInference


class TestExportOnnx(ExtTestCase):

    def verify(self, content):
        try:
            left, __ = verify_code(content, exc=False)
        except SyntaxError as e:
            raise AssertionError(
                "Unable to analyse a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e

        # execution
        try:
            obj = compile(content, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e
        glo = globals().copy()
        loc = {'numpy_helper': numpy_helper,
               'make_model': make_model,
               'make_node': make_node,
               'set_model_props': set_model_props,
               'make_tensor': make_tensor,
               'make_graph': make_graph,
               'make_tensor_value_info': make_tensor_value_info,
               'print': print, 'sorted': sorted,
               'collections': collections, 'inspect': inspect}
        out = StringIO()
        err = StringIO()
        if len(left) >= 5:
            raise AssertionError(
                "Too many unknown symbols: %r." % left)

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              content)) from e
        return glo, loc

    def test_export_onnx(self):
        this = os.path.dirname(__file__)
        folder = os.path.join(this, "data")
        names = ["fft2d_any.onnx"]
        for name in names:
            with self.subTest(name=name):
                oinf0 = OnnxInference(os.path.join(folder, name))

                x = numpy.random.randn(3, 1, 4).astype(numpy.float32)
                y = oinf0.run({'x': x})

                new_onnx = export2onnx(
                    os.path.join(folder, name), name="FFT2D")
                _, loc = self.verify(new_onnx)
                model = loc['onnx_model']
                oinf = OnnxInference(model)
                y1 = oinf.run({'x': x})

                new_onnx = export2onnx(
                    os.path.join(folder, name), verbose=False)
                _, loc = self.verify(new_onnx)
                model = loc['onnx_model']
                oinf = OnnxInference(model)
                y2 = oinf.run({'x': x})

                self.assertEqualArray(y['y'], y1['y'])
                self.assertEqualArray(y['y'], y2['y'])

    def verify_tf(self, content):
        try:
            left, __ = verify_code(content, exc=False)
        except SyntaxError as e:
            raise AssertionError(
                "Unable to analyse a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e

        # execution
        try:
            obj = compile(content, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, content)) from e
        glo = globals().copy()
        loc = {'numpy': numpy, 'print': print,
               'dict': dict, 'sorted': sorted, 'list': list,
               'print': print, 'sorted': sorted,
               'collections': collections, 'inspect': inspect,
               'helper': helper}
        out = StringIO()
        err = StringIO()
        if len(left) >= 14:
            raise AssertionError(
                "Too many unknown symbols: %r." % left)

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              content)) from e
        return glo, loc

    def test_export2tf2onnx(self):
        this = os.path.dirname(__file__)
        folder = os.path.join(this, "data")
        names = ["fft2d_any.onnx"]
        for name in names:
            with self.subTest(name=name):
                new_onnx = export2tf2onnx(
                    os.path.join(folder, name), name="FFT2D")
                _, loc = self.verify_tf(new_onnx)
                model = loc['onnx_model']
                self.assertIn('op_type: "FFT2D"', str(model))
                # print(model)


if __name__ == "__main__":
    unittest.main()
