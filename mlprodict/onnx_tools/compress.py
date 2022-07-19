"""
@file
@brief Functions to simplify, compress an ONNX graph.

.. versionadded:: 0.9
"""
import logging
from onnx import ModelProto, GraphProto, FunctionProto
from onnx.helper import (
    make_function, make_model, make_value_info, make_graph,
    make_tensor_type_proto, make_node, make_operatorsetid)


logger = logging.getLogger('onnx:compress')


def _check_expression(expe):
    att = expe.attribute[0].g
    inputs = [i.name for i in att.input]
    if list(expe.input) != inputs:
        raise RuntimeError(  # pragma: no cover
            f'Name mismatch in node Expression {expe.input!r} != {inputs!r}.')
    outputs = [o.name for o in att.output]
    if list(expe.output) != outputs:
        raise RuntimeError(  # pragma: no cover
            f'Name mismatch in node Expression {expe.input!r} != {inputs!r}.')


def _fuse_node(o, node, node_next):
    """
    Merges two nodes having one input/output in common.

    :param o: output name
    :param node: first node (it outputs the results)
    :param node_next: second node (it ingests the result)
    :return: merged node
    """
    type_expression = ('mlprodict', 'Expression')
    if list(node.output) != [o]:
        raise RuntimeError(  # pragma: no cover
            f"The only output of the first node should be {[o]!r} not {node.output!r}.")
    cannot_do = {('', 'If'), ('', 'Loop'), ('', 'Scan')}
    key1 = node.domain, node.op_type
    if key1 in cannot_do:
        return None
    key2 = node_next.domain, node_next.op_type
    if key2 in cannot_do:
        return None

    if key1 == type_expression:
        _check_expression(node)
    if key2 == type_expression:
        _check_expression(node_next)

    graph = None

    if node.domain == '' and node_next.domain == '':
        # Simple case
        inputs = [make_value_info(name, make_tensor_type_proto(0, []))
                  for name in node.input]
        outputs = [make_value_info(name, make_tensor_type_proto(0, []))
                   for name in node_next.output]
        graph = make_graph([node, node_next], "expression", inputs, outputs)

    elif key1 == type_expression and node_next.domain == '':
        att = node.attribute[0].g
        inputs = att.input
        outputs = [make_value_info(name, make_tensor_type_proto(0, []))
                   for name in node_next.output]
        graph = make_graph(list(att.node) + [node_next],
                           "expression", inputs, outputs)

    elif node.domain == '' and key2 == type_expression:
        att = node_next.attribute[0].g
        inputs = [make_value_info(name, make_tensor_type_proto(0, []))
                  for name in node.input]
        outputs = att.output
        graph = make_graph([node] + list(att.node),
                           "expression", inputs, outputs)

    elif key1 == type_expression and key2 == type_expression:
        att1 = node.attribute[0].g
        att2 = node_next.attribute[0].g
        inputs = att1.input
        outputs = att2.output
        graph = make_graph(list(att1.node) + list(att2.node),
                           "expression", inputs, outputs)

    if graph is not None:
        new_node = make_node(
            'Expression', node.input, node_next.output, domain='mlprodict',
            expression=graph)
        return new_node

    raise NotImplementedError(  # pragma: no cover
        "Unable to merge nodes '%s/%s' and '%s/%s'." % (
            node.domain, node.op_type, node_next.domain, node_next.op_type))


def _compress_nodes_once(nodes, verbose=0):
    """
    Compresses a sequence of node to make it more
    readable. If possible, it creates a node `Expression`
    with a graph as an attribute.

    :param nodes: sequence of nodes to compress
    :return: compressed sequence of nodes
    """
    # check that a result is used only once
    order = {}
    results = {}
    for node in nodes:
        order[id(node)] = (len(order), node)
        for name in node.input:
            if name in results:
                results[name] += 1
            else:
                results[name] = 1

    once = {k: v for k, v in results.items() if v == 1}
    if len(once) == 0:
        return nodes

    once_nodes_o = {}
    once_nodes_i = {}
    for node in nodes:
        if len(node.output) != 1:
            continue
        for o in node.output:
            if o in once:
                once_nodes_o[o] = node
        for i in node.input:
            if i in once:
                once_nodes_i[i] = node

    if len(once_nodes_o) == 0:
        return nodes

    if verbose > 0:
        logger.debug(
            "Results to compress: %r", list(sorted(once_nodes_o)))

    while len(once_nodes_o) > 0:
        o, node = once_nodes_o.popitem()
        node_next = once_nodes_i[o]
        new_node = _fuse_node(o, node, node_next)
        if new_node is None:
            # nothing can be done
            continue
        once_nodes_o.update({o: new_node for o in node_next.output
                             if o in once_nodes_o})
        once_nodes_i.update({i: new_node for i in node.input
                             if i in once_nodes_i})
        order[id(new_node)] = (order[id(node)][0], new_node)
        del order[id(node)]
        del order[id(node_next)]

    ordered = list(sorted((v[0], k, v[1]) for k, v in order.items()))
    return [v[-1] for v in ordered]


def _compress_nodes(nodes, verbose=0):
    """
    Compresses a sequence of node to make it more
    readable. If possible, it creates a node `Expression`
    with a graph as an attribute.

    :param nodes: sequence of nodes to compress
    :return: compressed sequence of nodes
    """
    return _compress_nodes_once(nodes, verbose=verbose)


def compress_proto(proto, verbose=0):
    """
    Compresses a :epkg:`ModelProto`, :epkg:`FunctionProto`,
    :epkg:`GraphProto`. The function detects nodes outputting
    results only used once. It then fuses it with the node
    taking it as an input.

    :param proto: :epkg:`ModelProto`, :epkg:`FunctionProto`,
        :epkg:`GraphProto`
    :param verbose: logging
    :return: same type

    .. versionadded:: 0.9
    """
    if isinstance(proto, FunctionProto):
        nodes = _compress_nodes(proto.node, verbose=verbose)
        if len(nodes) == len(proto.node):
            # unchanged
            return proto
        if verbose:
            logger.debug(
                "Compressed function %r/%r from %d nodes to %d.",
                proto.domain, proto.name, len(proto.node), len(nodes))
        opsets = {op.domain: op.version for op in proto.opset_import}
        opsets['mlprodict'] = 1

        return make_function(
            proto.domain, proto.name,
            proto.input, proto.output, nodes,
            opset_imports=[
                make_operatorsetid(k, v) for k, v in opsets.items()],
            attributes=proto.attribute,
            doc_string=proto.doc_string)

    if isinstance(proto, ModelProto):
        modified = 0
        new_graph = compress_proto(proto.graph, verbose=verbose)
        if id(new_graph) != id(proto.graph):
            modified += 1
        fcts = []
        for f in proto.functions:
            new_f = compress_proto(f, verbose=verbose)
            if id(new_f) != id(f):
                modified += 1
            fcts.append(new_f)
        if modified == 0:
            return proto
        opsets = {op.domain: op.version for op in proto.opset_import}
        opsets['mlprodict'] = 1
        if verbose:
            logger.debug(
                "Compressed model %s modified=%d.", proto.name, modified)
        return make_model(
            new_graph, functions=fcts,
            opset_imports=[
                make_operatorsetid(k, v) for k, v in opsets.items()],
            producer_name=proto.producer_name,
            producer_version=proto.producer_version,
            ir_version=proto.ir_version,
            doc_string=proto.doc_string,
            domain=proto.domain,
            model_version=proto.model_version)

    if isinstance(proto, GraphProto):
        nodes = _compress_nodes(proto.node, verbose=verbose)
        if len(nodes) == len(proto.node):
            # unchanged
            return proto
        if verbose:
            logger.debug(
                "Compressed graph %s from %d nodes to %d.",
                proto.name, len(proto.node), len(nodes))
        return make_graph(
            nodes, proto.name, proto.input, proto.output,
            proto.initializer, sparse_initializer=proto.sparse_initializer)

    raise TypeError(  # pragma: no cover
        "Unexpected type for proto %r, it should ModelProto, "
        "GraphProto or FunctionProto." % type(proto))
