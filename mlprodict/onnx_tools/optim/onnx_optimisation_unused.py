"""
@file
@brief Optimisation of :epkg:`ONNX` graphs.
"""
import logging
from onnx import FunctionProto, GraphProto
from onnx.helper import make_graph, make_function
from ._onnx_optimisation_common import (  # pylint: disable=E0611
    _apply_optimisation_on_graph, _apply_remove_node_fct_node)


logger = logging.getLogger('onnx:optim')


def _process_node(node, data, edges, paths, prefix="", sep="::", path=None):
    node_name = prefix + node.name
    data[node_name, 1] = node
    path = [] if path is None else path.copy()
    paths[node_name, 1] = path
    path = path.copy()
    path.append(node_name)
    for inp in node.input:
        data[inp, 0] = node
        edges[(inp, 0), (node_name, 1)] = node
        paths[inp, 0] = path
        if '::' in node_name:
            # We need to link an input to the parent node
            # if the node is part of subgraph.
            # path_r = paths[inp, 0]
            if len(path) <= 1:
                raise RuntimeError(  # pragma: no cover
                    f"Unexpected path {path!r}.")
            edges[(inp, 0), (path[-2], 1)] = node

    for out in node.output:
        data[out, 0] = node
        paths[out, 0] = node_name
        edges[(node_name, 1), (out, 0)] = node
    if len(node.attribute) > 0:
        for att in node.attribute:
            if not hasattr(att, 'g'):
                continue
            if not isinstance(att.g, GraphProto):
                continue
            for no in att.g.node:
                _process_node(no, data, edges, paths,
                              prefix=node_name + sep, path=path)


def onnx_remove_node_unused(onnx_model, recursive=True, debug_info=None, **options):
    """
    Removes unused nodes of the graph. An unused node
    is not involved in the output computation.

    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :param debug_info: debug information (private)
    :param options: unused
    :return: new onnx _model
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
    logger.debug("onnx_remove_node_unused:begin with %d nodes.",
                 len(graph.node))
    is_function = isinstance(graph, FunctionProto)
    data = {}
    valid = {}
    edges = {}
    paths = {}

    if not is_function:
        for init in graph.initializer:
            data[init.name, 0] = init

    for node in graph.node:
        _process_node(node, data, edges, paths)

    for out in graph.output:
        valid[out if is_function else out.name, 0] = True

    modif = 1
    while modif > 0:
        modif = 0
        for e1, e2 in edges:  # pylint: disable=E1141
            if valid.get(e2, False) and not valid.get(e1, False):
                valid[e1] = True
                modif += 1

    new_nodes = [n for n in graph.node if (n.name, 1) in valid]
    if not is_function:
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
    if is_function:
        logger.debug("onnx_remove_node_unused:end function with %d nodes.",
                     len(nodes))
        return make_function(
            onnx_model.domain, onnx_model.name,
            onnx_model.input, onnx_model.output, nodes,
            opset_imports=onnx_model.opset_import,
            attributes=onnx_model.attribute,
            doc_string=onnx_model.doc_string)
    graph = make_graph(nodes, onnx_model.name,
                       onnx_model.input, onnx_model.output,
                       new_inits)
    graph.value_info.extend(onnx_model.value_info)  # pylint: disable=E1101
    logger.debug("onnx_remove_node_unused:end graph with %d nodes.",
                 len(nodes))
    return graph
