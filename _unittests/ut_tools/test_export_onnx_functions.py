"""
@brief      test log(time=14s)
"""
import collections
import inspect
import unittest
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import numpy
from onnx import numpy_helper
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor, make_graph,
    make_tensor_value_info, make_opsetid, make_function)
from pyquickhelper.pycode import ExtTestCase
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.tools.code_helper import print_code
from mlprodict.onnx_tools.onnx_export import (
    export2onnx, export2xop)
from mlprodict.testing.verify_code import verify_code
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx
from mlprodict.npy.xop_variable import Variable
from mlprodict.npy.xop import loadop, OnnxOperatorFunction


class TestExportOnnxFunction(ExtTestCase):

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
                "" % (e, print_code(content))) from e
        glo = globals().copy()
        loc = {'numpy_helper': numpy_helper,
               'make_model': make_model,
               'make_node': make_node,
               'set_model_props': set_model_props,
               'make_tensor': make_tensor,
               'make_graph': make_graph,
               'make_function': make_function,
               'make_tensor_value_info': make_tensor_value_info,
               'print': print, 'sorted': sorted,
               'make_opsetid': make_opsetid,
               'Variable': Variable, 'loadop': loadop,
               'OnnxOperatorFunction': OnnxOperatorFunction,
               'collections': collections, 'inspect': inspect}
        out, err = StringIO(), StringIO()
        if len(left) >= 10:
            raise AssertionError(
                f"Too many unknown symbols: {left!r} in\n{content}")

        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(obj, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              print_code(content))) from e
        return glo, loc

    def test_pipeline_pipeline_function(self):
        x = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=numpy.float32)
        model = Pipeline([
            ("pipe1", Pipeline(
                [('sub1', StandardScaler()), ('sub2', StandardScaler())])),
            ("scaler2", StandardScaler())])
        model.fit(x)
        model_onnx = to_onnx(
            model, initial_types=[("X", FloatTensorType([None, 2]))],
            as_function=True, target_opset=15)
        self.assertGreater(len(model_onnx.functions), 1)
        rt = 'python'
        oinf0 = OnnxInference(model_onnx, runtime=rt)
        y0 = oinf0.run({'X': x})

        new_onnx_code = export2onnx(model_onnx, name="function")
        self.assertIn('make_function', new_onnx_code)
        _, loc = self.verify(new_onnx_code)
        model = loc['onnx_model']
        oinf1 = OnnxInference(model, runtime=rt)
        y1 = oinf1.run({'X': x})
        self.assertEqualArray(y0['main_scaler2_variable'],
                              y1['main_scaler2_variable'])

        new_onnx_code = export2xop(model_onnx, name="function")
        _, loc = self.verify(new_onnx_code)
        model = loc['onnx_model']
        self.assertEqual(len(model_onnx.functions), len(model.functions))
        oinf1 = OnnxInference(model, runtime=rt)
        y1 = oinf1.run({'X': x})
        self.assertEqualArray(y0['main_scaler2_variable'],
                              y1['main_scaler2_variable'])


if __name__ == "__main__":
    # TestExportOnnxFunction().test_export_function_onnx()
    unittest.main(verbosity=2)
