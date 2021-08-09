# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import os
import unittest
import numpy
from onnx import helper, TensorProto
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools.graphs import onnx_graph_distance


class TestGraphsDistance(ExtTestCase):


    def test_bug_graph_infinite(self):
        from mlstatpy.graph.graphviz_helper import draw_graph_graphviz

        shape = None
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, shape)
        Z = helper.make_tensor_value_info('Z', TensorProto.INT64, shape)
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


if __name__ == "__main__":
    unittest.main()
