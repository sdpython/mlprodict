"""
@brief      test log(time=3s)
"""
import os
import unittest
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import numpy
from onnx import numpy_helper
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor, make_graph,
    make_tensor_value_info)
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_tools.onnx_export import export2onnx, export2tf2onnx
from mlprodict.testing.verify_code import verify_code
from mlprodict.onnxrt import OnnxInference


class TestExportOnnx(ExtTestCase):
    
    def verify(self, content, existing_loc=None):
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
               'make_tensor_value_info': make_tensor_value_info}
        if existing_loc is not None:
            loc.update(existing_loc)
            glo.update(existing_loc)
        out = StringIO()
        err = StringIO()
        self.assertLess(len(left), 5)

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
                
                new_onnx = export2onnx(os.path.join(folder, name))
                glo, loc = self.verify(new_onnx)
                model = loc['onnx_model']
                oinf = OnnxInference(model)
                y1 = oinf0.run({'x': x})

                new_onnx = export2onnx(os.path.join(folder, name), verbose=False)
                glo, loc = self.verify(new_onnx)
                model = loc['onnx_model']
                oinf = OnnxInference(model)
                y2 = oinf0.run({'x': x})
                
                self.assertEqualArray(y['y'], y1['y'])
                self.assertEqualArray(y['y'], y2['y'])

    def verify_tf(self, content, existing_loc=None):
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
               'make_tensor_value_info': make_tensor_value_info}
        if existing_loc is not None:
            loc.update(existing_loc)
            glo.update(existing_loc)
        out = StringIO()
        err = StringIO()
        self.assertLess(len(left), 5)

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
                new_onnx = export2tf2onnx(os.path.join(folder, name))
                print(new_onnx)
                self.verify_tf(new_onnx)




if __name__ == "__main__":
    unittest.main()
