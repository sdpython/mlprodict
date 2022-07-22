# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import os
import unittest
import numpy
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxSub  # pylint: disable=E0611
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.tools.graphs import onnx2bigraph, BiGraph


class TestGraphs(ExtTestCase):

    def fit(self, model):
        data = load_iris()
        X, y = data.data, data.target
        model.fit(X, y)
        return model

    def test_exc(self):
        self.assertRaise(lambda: BiGraph([], [], []), TypeError)
        self.assertRaise(lambda: BiGraph({}, [], []), TypeError)
        self.assertRaise(lambda: BiGraph({}, {}, []), TypeError)
        self.assertRaise(
            lambda: BiGraph({'a': None}, {'b': None}, {('a', 'a'): None}),
            ValueError)
        self.assertRaise(
            lambda: BiGraph({'a': None}, {'a': None}, {('a', 'a'): None}),
            ValueError)

    def test_pipe_graph(self):
        model = self.fit(
            make_pipeline(StandardScaler(), LogisticRegression()))
        onx = to_onnx(model, numpy.zeros((3, 4), dtype=numpy.float64))
        bigraph = onnx2bigraph(onx)
        text = str(bigraph)
        self.assertEqual(text, "BiGraph(19 v., 12 v., 30 edges)")
        obj = list(bigraph)
        self.assertEqual(len(obj), 61)
        for o in obj:
            self.assertEqual(len(o), 3)
            self.assertIn(o[0], {-1, 0, 1})
            self.assertIsInstance(o[1], (str, tuple))
            self.assertStartsWith("A(", str(o[-1]))

    def test_pipe_graph_order(self):
        model = self.fit(
            make_pipeline(StandardScaler(), LogisticRegression()))
        onx = to_onnx(model, numpy.zeros((3, 4), dtype=numpy.float64))
        bigraph = onnx2bigraph(onx)
        order = bigraph.order_vertices()
        self.assertEqual(len(order), 31)
        self.assertIsInstance(order, dict)
        for k in order:
            self.assertIsInstance(bigraph[k], BiGraph.A)
        ed = list(bigraph.edges)[0]
        self.assertIsInstance(bigraph[ed], BiGraph.A)

    def test_pipe_graph_display(self):
        model = self.fit(
            make_pipeline(StandardScaler(), LogisticRegression()))
        onx = to_onnx(model, numpy.zeros((3, 4), dtype=numpy.float64))
        bigraph = onnx2bigraph(onx)
        graph = bigraph.display_structure()
        text = str(graph)
        self.assertIn("AdjacencyGraphDisplay(", text)
        self.assertIn("Action(", text)

    def test_pipe_graph_display_text(self):
        idi = numpy.identity(2).astype(numpy.float32)
        opv = TARGET_OPSET
        A = OnnxAdd('X', idi, op_version=opv)
        B = OnnxSub(A, 'W', output_names=['Y'], op_version=opv)
        onx = B.to_onnx({'X': idi.astype(numpy.float32),
                         'W': idi.astype(numpy.float32)})
        bigraph = onnx2bigraph(onx)
        graph = bigraph.display_structure()
        text = graph.to_text()
        for c in ['Input-1', 'Input-0', 'Output-0', 'W', 'W', 'I0', 'I1',
                  'inout', 'O0 I0', 'A  S']:
            self.assertIn(c, text)

    def test_bug_graph(self):
        this = os.path.abspath(os.path.dirname(__file__))
        data = os.path.join(this, "data", "bug_graph.onnx")
        oinf = OnnxInference(
            data, inside_loop=True,
            static_inputs=['StatefulPartitionedCall/Reshape:0'])
        text = oinf.to_text(distance=8)
        self.assertIn(
            "cond___pcen/simple_rnn/while/Identity_graph_outputs_Identity__4:0",
            text)

    def test_bug_graph_infinite(self):
        this = os.path.abspath(os.path.dirname(__file__))
        data = os.path.join(this, "data", "bug_graph_infinite.onnx")
        oinf = OnnxInference(data, inside_loop=True)
        text = oinf.to_text(distance=8)
        self.assertIn("slice_end", text)

    def test_pipe_graph_simplified(self):
        model = self.fit(
            make_pipeline(StandardScaler(), LogisticRegression()))
        onx = to_onnx(model, numpy.zeros((3, 4), dtype=numpy.float64))
        bigraph = onnx2bigraph(onx, graph_type='simplified')
        text = str(bigraph)
        self.assertEqual(text, "BiGraph(19 v., 12 v., 30 edges)")
        disp = bigraph.summarize()
        self.assertIn("B('Cast', '042434366f', 'Cast1')", disp)
        self.assertIn("B('Div', '', 'Di_Div'", disp)


if __name__ == "__main__":
    unittest.main()
