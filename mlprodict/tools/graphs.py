"""
@file
@brief Alternative to dot to display a graph.

.. versionadded:: 0.7
"""
import os
import urllib.request
from collections import OrderedDict
import numpy


class BiGraph:
    """
    BiGraph representation.
    """

    class A:
        "Additional information for a vertex or an edge."

        def __init__(self, kind):
            self.kind = kind

        def __repr__(self):
            return "A(%r)" % self.kind

    def __init__(self, v0, v1, edges):
        """
        :param v0: first set of vertices (dictionary)
        :param v1: second set of vertices (dictionary)
        :param edges: edges
        """
        if not isinstance(v0, dict):
            raise TypeError("v0 must be a dictionary.")
        if not isinstance(v1, dict):
            raise TypeError("v0 must be a dictionary.")
        if not isinstance(edges, dict):
            raise TypeError("edges must be a dictionary.")
        self.v0 = v0
        self.v1 = v1
        self.edges = edges
        common = set(self.v0).intersection(set(self.v1))
        if len(common) > 0:
            raise ValueError(
                "Sets v1 and v2 have common nodes (forbidden): %r." % common)
        for a, b in edges:
            if a in v0 and b in v1:
                continue
            if a in v1 and b in v0:
                continue
            raise ValueError(
                "Edges (%r, %r) not found among the vertices." % (a, b))

    def __str__(self):
        """
        usual
        """
        return "%s(%d v., %d v., %d edges)" % (
            self.__class__.__name__, len(self.v0),
            len(self.v1), len(self.edges))

    def __iter__(self):
        """
        Iterates over all vertices and edges.
        It produces 3-uples,
        * 0, name, A: vertices in *v0*
        * 1, name, A: vertices in *v1*
        * -1, name, A: edges
        """
        for k, v in self.v0.items():
            yield 0, k, v
        for k, v in self.v1.items():
            yield 1, k, v
        for k, v in self.edges.items():
            yield -1, k, v

    def __getitem__(self, key):
        """
        Returns a vertex is key is a string or an edge
        if it is a tuple.

        :param key: vertex or edge name
        :return: value
        """
        if isinstance(key, tuple):
            return self.edges[key]
        if key in self.v0:
            return self.v0[key]
        return self.v1[key]

    def order_vertices(self):
        """
        Orders the vertices from the input to the output.

        :return: dictionary `{vertex name: order}`
        """
        order = {}
        for v in self.v0:
            order[v] = 0
        for v in self.v1:
            order[v] = 0
        modif = 1
        while modif > 0:
            modif = 0
            for a, b in self.edges:
                if order[b] <= order[a]:
                    order[b] = order[a] + 1
                    modif += 1
        return order


def onnx2bigraph(model_onnx, recursive=False):
    """
    Converts an ONNX graph into a graph representation,
    edges, vertices.

    :param model_onnx: ONNX graph
    :param recursive: dig into subgraphs too
    :return: see @cl BiGraph
    """
    if recursive:
        raise NotImplementedError(  # pragma: no cover
            "Option recursive=True is not implemented yet.")
    v0 = {}
    v1 = {}
    edges = {}

    # inputs
    for o in model_onnx.graph.input:
        v0[o.name] = BiGraph.A('I')
    for o in model_onnx.graph.output:
        v0[o.name] = BiGraph.A('O')
    for o in model_onnx.graph.initializer:
        v0[o.name] = BiGraph.A('Init')
    for n in model_onnx.graph.node:
        v1[n.name] = BiGraph.A(n.op_type)
        for o in n.input:
            edges[o, n.name] = BiGraph.A('I')
        for o in n.output:
            if o not in v0:
                v0[o] = BiGraph.A('o')
            edges[n.name, o] = BiGraph.A('O')

    return BiGraph(v0, v1, edges)
