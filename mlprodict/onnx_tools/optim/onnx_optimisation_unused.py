"""
@file
@brief Optimisation of :epkg:`ONNX` graphs.
"""
from onnx.helper import make_graph
from ._onnx_optimisation_common import (  # pylint: disable=E0611
    _apply_optimisation_on_graph, _apply_remove_node_fct_node)


def onnx_remove_node_unused(onnx_model, recursive=True, debug_info=None, **options):
    """
    Removes unused nodes of the graph. An unused node
    is not involved in the output computation.

    @param      onnx_model      onnx model
    @param      recursive       looks into subgraphs
    @param      debug_info      debug information (private)
    @param      options         unused
    @return                     new onnx _model
    """
    if debug_info is None:
        debug_info = [str(type(onnx_model)).rsplit(
            '.', maxsplit=1)[-1].strip("'>")]
    else:
        debug_info = (debug_info +
                      [str(type(onnx_model)).rsplit('.', maxsplit=1)[-1].strip("'>")])

    if hasattr(onnx_model, 'graph'):
        return _apply_optimisation_on_graph(
            onnx_remove_node_unused, onnx_model,
            recursive=recursive, debug_info=debug_info,
            **options)

    graph = onnx_model
    data = {}
    valid = {}
    edges = {}

    for init in graph.initializer:
        data[init.name, 0] = init

    for node in graph.node:
        data[node.name, 1] = node
        for inp in node.input:
            data[inp, 0] = node
            edges[(inp, 0), (node.name, 1)] = node
        for out in node.output:
            data[out, 0] = node
            edges[(node.name, 1), (out, 0)] = node

    for out in graph.output:
        valid[out.name, 0] = True

    modif = 1
    while modif > 0:
        modif = 0
        for e1, e2 in edges:  # pylint: disable=E1141
            if valid.get(e2, False) and not valid.get(e1, False):
                valid[e1] = True
                modif += 1

    new_nodes = [n for n in graph.node if (n.name, 1) in valid]
    new_inits = [n for n in graph.initializer if (n.name, 0) in valid]

    if recursive:
        # Handles subgraphs.
        for i in range(len(new_nodes)):  # pylint: disable=C0200
            node = new_nodes[i]
            if node is None or not (node.attribute):  # pylint: disable=C0325
                continue
            new_nodes[i] = _apply_remove_node_fct_node(
                onnx_remove_node_unused,
                node, recursive=True, debug_info=debug_info + [node.name])

    # Finally create the new graph.
    nodes = list(filter(lambda n: n is not None, new_nodes))
    graph = make_graph(nodes, onnx_model.name,
                       onnx_model.input, onnx_model.output,
                       new_inits)

    graph.value_info.extend(onnx_model.value_info)  # pylint: disable=E1101
    return graph
