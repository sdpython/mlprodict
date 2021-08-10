# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import os
import unittest
import numpy
from onnx import helper, TensorProto
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools.graphs import onnx_graph_distance
from mlprodict.onnx_conv import to_onnx


class TestGraphsDistance(ExtTestCase):

    def test_graph_distance(self):
        from mlstatpy.graph.graphviz_helper import draw_graph_graphviz

        shape = None
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, shape)  # pylint: disable=E1101
        Z = helper.make_tensor_value_info('Z', TensorProto.INT64, shape)  # pylint: disable=E1101
        node_def = helper.make_node('Shape', ['X'], ['Z'], name='Zt')
        graph_def = helper.make_graph([node_def], 'test-model', [X], [Z])
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict', ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 13)])

        d, graph = onnx_graph_distance(model_def, model_def)
        self.assertLess(d, 1)
        vertices, edges = graph.draw_vertices_edges()
        gv = draw_graph_graphviz(vertices, edges)
        self.assertIn("->", gv)

    def test_graph_distance_bigger(self):
        from mlstatpy.graph.graphviz_helper import draw_graph_graphviz

        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, None)  # pylint: disable=E1101
        Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, None)  # pylint: disable=E1101
        node_def = helper.make_node('Neg', ['X'], ['Z'], name='A')
        graph_def = helper.make_graph([node_def], 'test-model', [X], [Z])
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict', ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 13)])

        node_def1 = helper.make_node('Neg', ['X'], ['Y'], name='A')
        node_def2 = helper.make_node('Neg', ['Y'], ['Z'], name='B')
        graph_def = helper.make_graph(
            [node_def1, node_def2], 'test-model', [X], [Z])
        model_def2 = helper.make_model(
            graph_def, producer_name='mlprodict', ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 13)])

        d, graph = onnx_graph_distance(model_def, model_def2)
        self.assertLess(d, 1)
        vertices, edges = graph.draw_vertices_edges()
        gv = draw_graph_graphviz(vertices, edges)
        self.assertIn("->", gv)

    def test_graph_distance_profile(self):
        data = load_iris()
        X = data.data.astype(numpy.float32)
        model = KMeans(n_clusters=3)
        model.fit(X)
        model_onnx = to_onnx(model, X, target_opset=13)
        with open("temp_kmeans.onnx", "wb") as f:
            f.write(model_onnx.SerializeToString())

        rootrem = os.path.normpath(os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "..", "..", ".."))
        res = self.profile(
            lambda: onnx_graph_distance(
                model_onnx, model_onnx, verbose=1),
            rootrem=rootrem)
        if __name__ == "__main__":
            print(res[1])
        self.assertIn("cumtime", res[1])


if __name__ == "__main__":
    unittest.main()
