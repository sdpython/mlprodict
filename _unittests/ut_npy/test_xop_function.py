# pylint: disable=E0611
"""
@brief      test log(time=15s)
"""
import unittest
import numpy
from onnx import TensorProto, AttributeProto
from onnx.helper import (  # pylint: disable=W0611
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid,
    make_function)
from pyquickhelper.pycode import ExtTestCase
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop, OnnxOperatorFunction
from mlprodict.onnx_tools.onnx_manipulations import (
    onnx_model_to_function, get_opsets)
from mlprodict.onnx_tools.model_checker import check_onnx


class TestXOpsFunction(ExtTestCase):

    def test_onnx_function_init(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')
        fct = OnnxOperatorFunction(proto, 'X')
        rp = repr(fct)
        self.assertStartsWith("OnnxOperatorFunction(", rp)
        op = OnnxDiv(fct, numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

    def test_onnx_function_to_python(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')
        op = OnnxDiv(OnnxOperatorFunction(proto, 'X'),
                     numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx, runtime='python')
        py = oinf.to_python()
        items = list(py.items())
        value = items[0][1]
        self.assertIn('return OnnxPythonInference().run(X)', value)
        self.assertIn('def pyrt_mlprodict_AddAbs(X):', value)

    def test_onnx_function_init_identity(self):
        OnnxAbs, OnnxAdd, OnnxDiv, OnnxIdentity = loadop(
            "Abs", "Add", "Div", "Identity")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')
        op = OnnxDiv(OnnxOperatorFunction(proto, OnnxIdentity('X')),
                     numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

    def test_onnx_function(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        op = OnnxDiv(ad('X'), numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

    def test_onnx_function_initializer(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd(ov, numpy.array([1], dtype=numpy.float32),
                     output_names=['Y'])
        op = OnnxDiv(ad('X'), numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((numpy.abs(x) + 1) / 2, got['Y'])

    def test_onnx_function_name(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('XX')
        ad = OnnxAdd('XX', ov)
        op = OnnxDiv(ad, numpy.array([2], dtype=numpy.float32),
                     output_names=['YY'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'XX': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['YY'])

        fonx, _ = onnx_model_to_function(onx, domain='sklearn')
        fct = OnnxOperatorFunction(fonx, 'X', output_names=['Y'])
        onx2 = fct.to_onnx(numpy.float32, numpy.float32)
        oinf = OnnxInference(onx2)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

        opsets = get_opsets(fonx)
        self.assertEqual(len(opsets), 1)

    def test_onnx_function_name2(self):
        OnnxAbs, OnnxAdd = loadop("Abs", "Add")
        ov = OnnxAbs('XX')
        ad = OnnxAdd('XX', ov, output_names=['YY'])
        onx = ad.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'XX': x})
        self.assertEqualArray(x + numpy.abs(x), got['YY'])

        fonx, _ = onnx_model_to_function(onx, domain='sklearn')
        fct1 = OnnxOperatorFunction(fonx, 'X')
        fct = OnnxOperatorFunction(fonx, fct1, output_names=['Y'])
        onx2 = fct.to_onnx(numpy.float32, numpy.float32)
        oinf = OnnxInference(onx2)
        x = numpy.array([-2, 3], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) * 2, got['Y'])

    def test_onnx_function_att_plot(self):

        new_domain = 'custom'
        opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

        cst = make_node('Constant', [], ['B'])
        att = AttributeProto()
        att.name = "value"
        att.ref_attr_name = "bias"
        att.type = AttributeProto.TENSOR
        cst.attribute.append(att)

        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])

        linear_regression = make_function(
            new_domain, 'LinearRegression', ['X', 'A'],
            ['Y'], [cst, node1, node2], opset_imports,
            ["bias"])

        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

        graph = make_graph(
            [make_node('LinearRegression', ['X', 'A'], ['Y1'], domain=new_domain,
                       bias=make_tensor('former_B', TensorProto.FLOAT, [1], [0.67])),
             make_node('Abs', ['Y1'], ['Y'])],
            'example',
            [X, A], [Y])

        onnx_model = make_model(
            graph, opset_imports=opset_imports,
            functions=[linear_regression])
        check_onnx(onnx_model)

        text = onnx_simple_text_plot(onnx_model)
        self.assertIn("attribute: 'bias'", text)
        self.assertIn("Constant(value=$bias)", text)
        self.assertIn("LinearRegression[custom](X, A, bias=[0.670000", text)

    def test_onnx_function_att_execute(self):

        new_domain = 'custom'
        opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

        cst = make_node('Constant', [], ['B'])
        att = AttributeProto()
        att.name = "value"
        att.ref_attr_name = "bias"
        att.type = AttributeProto.TENSOR
        cst.attribute.append(att)

        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])

        linear_regression = make_function(
            new_domain, 'LinearRegression', ['X', 'A'],
            ['Y'], [cst, node1, node2], opset_imports,
            ["bias"])

        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

        graph = make_graph(
            [make_node('LinearRegression', ['X', 'A'], ['Y1'], domain=new_domain,
                       bias=make_tensor('former_B', TensorProto.FLOAT, [1], [0.67])),
             make_node('Abs', ['Y1'], ['Y'])],
            'example',
            [X, A], [Y])

        onnx_model = make_model(
            graph, opset_imports=opset_imports,
            functions=[linear_regression])
        check_onnx(onnx_model)
        oinf = OnnxInference(onnx_model)
        x = numpy.array([[0, 1], [2, 3]], dtype=numpy.float32)
        a = numpy.array([[4, 5], [6, 7]], dtype=numpy.float32)

        def my_print(*args):
            pass

        exe2 = oinf.run({'X': x, 'A': a})
        exe = oinf.run({'X': x, 'A': a}, verbose=2, fLOG=my_print)
        self.assertEqualArray(exe['Y'], exe2['Y'])
        self.assertEqualArray(exe['Y'], x @ a + 0.67)

    def test_onnx_function_inside_function(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('XX')
        ad = OnnxAdd('XX', ov)
        op = OnnxDiv(ad, numpy.array([2], dtype=numpy.float32),
                     output_names=['YY'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        fonx, _ = onnx_model_to_function(onx, domain='sklearn', name='f1')
        fct = OnnxOperatorFunction(fonx, 'X', output_names=['Y'])

        onx2 = fct.to_onnx(numpy.float32, numpy.float32)
        fonx2, fps2 = onnx_model_to_function(onx2, domain='sklearn', name='f2')
        self.assertEqual(len(fps2), 1)
        fct2 = OnnxAdd(
            OnnxOperatorFunction(fonx2, 'X', sub_functions=fps2),
            numpy.array([1], dtype=numpy.float32),
            output_names=['Y'])
        onx3 = fct2.to_onnx(numpy.float32, numpy.float32)
        self.assertEqual(len(onx3.functions), 2)
        oinf = OnnxInference(onx3)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2 + 1, got['Y'])


if __name__ == "__main__":
    # TestXOpsFunction().test_onnx_function_att_execute()
    unittest.main(verbosity=2)
