# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import textwrap
import numpy
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_function,
    make_graph, make_tensor_value_info, make_opsetid)
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsRegressor
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxSub, OnnxDiv, OnnxAbs, OnnxLeakyRelu, OnnxGreater,
    OnnxReduceSum, OnnxIf)
from mlprodict.onnx_conv import to_onnx
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.plotting.plotting import (
    onnx_text_plot, onnx_text_plot_tree, onnx_simple_text_plot,
    onnx_text_plot_io)
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop_variable import Variable
from mlprodict.npy.xop import loadop, OnnxOperatorFunction


class TestPlotTextPlotting(ExtTestCase):

    def test_onnx_text_plot(self):
        idi = numpy.identity(2).astype(numpy.float32)
        opv = TARGET_OPSET
        A = OnnxAdd('X', idi, op_version=opv)
        B = OnnxSub(A, 'W', output_names=['Y'], op_version=opv)
        onx = B.to_onnx({'X': idi.astype(numpy.float32),
                         'W': idi.astype(numpy.float32)})
        res = onnx_text_plot(onx)
        self.assertIn("Init", res)

    def test_onnx_text_plot_tree(self):
        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        clr = DecisionTreeRegressor(max_depth=3)
        clr.fit(X, y)
        onx = to_onnx(clr, X)
        res = onnx_text_plot_tree(onx.graph.node[0])
        self.assertIn("treeid=0", res)
        self.assertIn("         T y=", res)

    @ignore_warnings(UserWarning)
    def test_onnx_simple_text_plot_kmeans(self):
        x = numpy.random.randn(10, 3)
        model = KMeans(3)
        model.fit(x)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_simple_text_plot(onx)
        expected1 = textwrap.dedent("""
        ReduceSumSquare(X, axes=[1], keepdims=1) -> Re_reduced0
          Mul(Re_reduced0, Mu_Mulcst) -> Mu_C0
            Gemm(X, Ge_Gemmcst, Mu_C0, alpha=-2.00, transB=1) -> Ge_Y0
          Add(Re_reduced0, Ge_Y0) -> Ad_C01
            Add(Ad_Addcst, Ad_C01) -> Ad_C0
              Sqrt(Ad_C0) -> scores
              ArgMin(Ad_C0, axis=1, keepdims=0) -> label
        """).strip(" \n")
        expected2 = textwrap.dedent("""
        ReduceSumSquare(X, axes=[1], keepdims=1) -> Re_reduced0
          Mul(Re_reduced0, Mu_Mulcst) -> Mu_C0
            Gemm(X, Ge_Gemmcst, Mu_C0, alpha=-2.00, transB=1) -> Ge_Y0
          Add(Re_reduced0, Ge_Y0) -> Ad_C01
            Add(Ad_Addcst, Ad_C01) -> Ad_C0
              Sqrt(Ad_C0) -> scores
              ArgMin(Ad_C0, axis=1, keepdims=0) -> label
        """).strip(" \n")
        expected3 = textwrap.dedent("""
        ReduceSumSquare(X, axes=[1], keepdims=1) -> Re_reduced0
          Mul(Re_reduced0, Mu_Mulcst) -> Mu_C0
            Gemm(X, Ge_Gemmcst, Mu_C0, alpha=-2.00, transB=1) -> Ge_Y0
          Add(Re_reduced0, Ge_Y0) -> Ad_C01
            Add(Ad_Addcst, Ad_C01) -> Ad_C0
              ArgMin(Ad_C0, axis=1, keepdims=0) -> label
              Sqrt(Ad_C0) -> scores
        """).strip(" \n")
        if (expected1 not in text and expected2 not in text and
                expected3 not in text):
            raise AssertionError(
                "Unexpected value:\n%s" % text)

    def test_onnx_simple_text_plot_knnr(self):
        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = RadiusNeighborsRegressor(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_simple_text_plot(onx, verbose=False)
        expected = "              Neg(arange_y0) -> arange_Y0"
        self.assertIn(expected, text)
        self.assertIn(", to=7)", text)
        self.assertIn(", keepdims=0)", text)
        self.assertIn(", perm=[1,0])", text)

    def test_onnx_simple_text_plot_toy(self):
        x = numpy.random.randn(10, 3).astype(numpy.float32)
        node1 = OnnxAdd('X', x, op_version=15)
        node2 = OnnxSub('X', x, op_version=15)
        node3 = OnnxAbs(node1, op_version=15)
        node4 = OnnxAbs(node2, op_version=15)
        node5 = OnnxDiv(node3, node4, op_version=15)
        node6 = OnnxAbs(node5, output_names=['Y'], op_version=15)
        onx = node6.to_onnx({'X': x.astype(numpy.float32)},
                            outputs={'Y': x}, target_opset=15)
        text = onnx_simple_text_plot(onx, verbose=False)
        expected = textwrap.dedent("""
        Add(X, Ad_Addcst) -> Ad_C0
          Abs(Ad_C0) -> Ab_Y0
        Identity(Ad_Addcst) -> Su_Subcst
          Sub(X, Su_Subcst) -> Su_C0
            Abs(Su_C0) -> Ab_Y02
            Div(Ab_Y0, Ab_Y02) -> Di_C0
              Abs(Di_C0) -> Y
        """).strip(" \n")
        self.assertIn(expected, text)
        text2, out, err = self.capture(
            lambda: onnx_simple_text_plot(onx, verbose=True))
        self.assertEqual(text, text2)
        self.assertIn('BEST:', out)
        self.assertEmpty(err)

    def test_onnx_simple_text_plot_leaky(self):
        x = OnnxLeakyRelu('X', alpha=0.5, op_version=15,
                          output_names=['Y'])
        onx = x.to_onnx({'X': FloatTensorType()},
                        outputs={'Y': FloatTensorType()},
                        target_opset=15)
        text = onnx_simple_text_plot(onx)
        expected = textwrap.dedent("""
        LeakyRelu(X, alpha=0.50) -> Y
        """).strip(" \n")
        self.assertIn(expected, text)

    def test_onnx_text_plot_io(self):
        x = OnnxLeakyRelu('X', alpha=0.5, op_version=15,
                          output_names=['Y'])
        onx = x.to_onnx({'X': FloatTensorType()},
                        outputs={'Y': FloatTensorType()},
                        target_opset=15)
        text = onnx_text_plot_io(onx)
        expected = textwrap.dedent("""
        input:
        """).strip(" \n")
        self.assertIn(expected, text)

    def test_onnx_simple_text_plot_if(self):

        opv = TARGET_OPSET
        x1 = numpy.array([[0, 3], [7, 0]], dtype=numpy.float32)
        x2 = numpy.array([[1, 0], [2, 0]], dtype=numpy.float32)

        node = OnnxAdd(
            'x1', 'x2', output_names=['absxythen'], op_version=opv)
        then_body = node.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxythen', FloatTensorType())])
        node = OnnxSub(
            'x1', 'x2', output_names=['absxyelse'], op_version=opv)
        else_body = node.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxyelse', FloatTensorType())])
        del else_body.graph.input[:]
        del then_body.graph.input[:]

        cond = OnnxGreater(
            OnnxReduceSum('x1', op_version=opv),
            OnnxReduceSum('x2', op_version=opv),
            op_version=opv)
        ifnode = OnnxIf(cond, then_branch=then_body.graph,
                        else_branch=else_body.graph,
                        op_version=opv, output_names=['y'])
        model_def = ifnode.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('y', FloatTensorType())])
        text = onnx_simple_text_plot(model_def)
        expected = textwrap.dedent("""
        input:
        """).strip(" \n")
        self.assertIn(expected, text)
        self.assertIn("If(Gr_C0) -> y", text)
        oinf = OnnxInference(model_def)
        text2 = oinf.to_text(kind="seq")
        self.assertEqual(text, text2)

    @ignore_warnings(UserWarning)
    def test_onnx_simple_text_plot_kmeans_links(self):
        x = numpy.random.randn(10, 3)
        model = KMeans(3)
        model.fit(x)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_simple_text_plot(onx, add_links=True)
        self.assertIn("Sqrt(Ad_C0) -> scores  <------", text)
        self.assertIn("|-|", text)

    def test_scan_plot(self):
        (OnnxSub, OnnxIdentity, OnnxReduceSumSquare, OnnxScan,  # pylint: disable=W0621
         OnnxAdd) = loadop('Sub', 'Identity',  # pylint: disable=W0621
                           'ReduceSumSquare', 'Scan', 'Add')

        def onnx_squareform_pdist(X, dtype=None, op_version=None, **kwargs):
            diff = OnnxSub('next_in', 'next',
                           op_version=op_version)
            id_next = OnnxIdentity('next_in', output_names=['next_out'],
                                   op_version=op_version)
            flat = OnnxReduceSumSquare(diff, axes=[1], op_version=op_version,
                                       output_names=['scan_out'], keepdims=0)
            scan_body = id_next.to_onnx(
                [Variable('next_in', numpy.float32, (None, None)),  # tensor_type([None, None])),
                 Variable('next', numpy.float32, (None, ))],  # tensor_type([None]))]),
                outputs=[Variable('next_out', numpy.float32, (None, None)),  # ([None, None])),
                         Variable('scan_out', numpy.float32, (None, ))],  # tensor_type([None]))],
                other_outputs=[flat],
                target_opset=op_version)
            node = OnnxScan(X, X, output_names=['S1', 'S2'],
                            num_scan_inputs=1,
                            body=(scan_body.graph, [id_next, flat]),
                            op_version=op_version, **kwargs)
            return node[1]

        cop = OnnxAdd('input', 'input')
        cdist = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': numpy.float32},
            outputs=[Variable('cdist', numpy.float32)])

        text = onnx_simple_text_plot(model_def, recursive=True)
        self.assertIn("----- subgraph", text)

    def test_function_plot(self):
        new_domain = 'custom'
        opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])

        linear_regression = make_function(
            new_domain,            # domain name
            'LinearRegression',     # function name
            ['X', 'A', 'B'],        # input names
            ['Y'],                  # output names
            [node1, node2],         # nodes
            opset_imports,          # opsets
            [])                     # attribute names

        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)

        graph = make_graph(
            [make_node('LinearRegression', ['X', 'A', 'B'], ['Y1'],
                       domain=new_domain),
             make_node('Abs', ['Y1'], ['Y'])],
            'example',
            [X, A, B], [Y])

        onnx_model = make_model(
            graph, opset_imports=opset_imports,
            functions=[linear_regression])  # functions to add)

        text = onnx_simple_text_plot(onnx_model)
        self.assertIn("function name=LinearRegression domain=custom", text)
        self.assertIn("MatMul(X, A) -> XA", text)
        self.assertIn("type=? shape=?", text)
        self.assertIn("LinearRegression[custom]", text)

    def test_onnx_function_init(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(  # pylint: disable=W0621
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')

        op = OnnxDiv(OnnxOperatorFunction(proto, 'X'),
                     numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        text = onnx_simple_text_plot(onx)
        self.assertIn("----- function name=AddAbs domain=mlprodict", text)


if __name__ == "__main__":
    unittest.main()
