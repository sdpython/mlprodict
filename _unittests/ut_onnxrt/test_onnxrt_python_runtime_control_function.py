"""
@brief      test log(time=2s)
"""
import unittest
import numpy
import onnx
from onnx import FunctionProto, parser
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.model_checker import check_onnx


class TestOnnxrtPythonRuntimeControlFunction(ExtTestCase):

    @ignore_warnings(DeprecationWarning)
    def test_if_function(self):

        then_out = onnx.helper.make_tensor_value_info(
            'then_out', onnx.TensorProto.FLOAT, [5])
        else_out = onnx.helper.make_tensor_value_info(
            'else_out', onnx.TensorProto.FLOAT, [5])

        x = numpy.array([1, 2, 3, 4, 5]).astype(numpy.float32)
        y = numpy.array([5, 4, 3, 2, 1]).astype(numpy.float32)

        then_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['then_out'],
            value=onnx.numpy_helper.from_array(x)
        )

        else_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['else_out'],
            value=onnx.numpy_helper.from_array(y)
        )

        then_body = onnx.helper.make_graph(
            [then_const_node],
            'then_body',
            [],
            [then_out]
        )

        else_body = onnx.helper.make_graph(
            [else_const_node],
            'else_body',
            [],
            [else_out]
        )

        if_node = onnx.helper.make_node(
            'If',
            inputs=['f_cond'],
            outputs=['f_res'],
            then_branch=then_body,
            else_branch=else_body
        )

        f = FunctionProto()
        f.domain = 'custom'
        f.name = 'fn'
        f.input.extend(['f_cond'])
        f.output.extend(['f_res'])
        f.node.extend([if_node])
        f.opset_import.extend([onnx.helper.make_opsetid("", 14)])

        graph = onnx.helper.make_graph(
            nodes=[onnx.helper.make_node('fn', domain='custom', inputs=[
                                         'cond'], outputs=['res'])],
            name='graph',
            inputs=[onnx.helper.make_tensor_value_info(
                'cond', onnx.TensorProto.BOOL, [])],
            outputs=[onnx.helper.make_tensor_value_info(
                'res', onnx.TensorProto.FLOAT, [5])],
        )

        m = onnx.helper.make_model(graph, producer_name='test',
                                   opset_imports=[onnx.helper.make_opsetid("", 14), onnx.helper.make_opsetid("custom", 1)])
        m.functions.extend([f])

        check_onnx(m)

        for rt in ['onnxruntime1', 'python']:
            with self.subTest(rt=rt):
                try:
                    oinf = OnnxInference(m.SerializeToString(), runtime=rt)
                except RuntimeError as e:
                    if "GraphProto attribute inferencing is not enabled" in str(e):
                        continue
                    raise e

                result = oinf.run({'cond': numpy.array(True)})
                expected = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
                self.assertEqualArray(expected, result['res'])

    @ignore_warnings(DeprecationWarning)
    def test_nested_local_functions(self):
        m = parser.parse_model('''
            <
              ir_version: 8,
              opset_import: [ "" : 14, "local" : 1],
              producer_name: "test",
              producer_version: "1.0",
              model_version: 1,
              doc_string: "Test preprocessing model"
            >
            agraph (uint8[H, W, C] x) => (uint8[H, W, C] x_processed)
            {
                x_processed = local.func(x)
            }

            <
              opset_import: [ "" : 14 ],
              domain: "local",
              doc_string: "function 1"
            >
            f1 (x) => (y) {
                y = Identity(x)
            }

            <
              opset_import: [ "" : 14 ],
              domain: "local",
              doc_string: "function 2"
            >
            f2 (x) => (y) {
                y = Identity(x)
            }

            <
              opset_import: [ "" : 14, "local" : 1 ],
              domain: "local",
              doc_string: "Preprocessing function."
            >
            func (x) => (y) {
                x1 = local.f1(x)
                y = local.f2(x1)
            }
        ''')

        text = onnx_simple_text_plot(m)
        self.assertIn("func[local](x) -> x_processed", text)
        check_onnx(m)

        for rt in ['python', 'onnxruntime1']:
            with self.subTest(rt=rt):
                try:
                    oinf = OnnxInference(m.SerializeToString(), runtime=rt)
                except RuntimeError as e:
                    if "func is not a registered function/op" in str(e):
                        continue
                    raise e

                x = numpy.array([0, 1, 3], dtype=numpy.uint8).reshape((1, 1, 3))
                result = oinf.run({'x': x})
                expected = x
                self.assertEqualArray(expected, result['x_processed'])


if __name__ == "__main__":
    unittest.main()
