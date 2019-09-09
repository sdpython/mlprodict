"""
@file
@brief Optimisation of :epkg:`ONNX` graphs.
"""
from onnx.helper import make_model, make_graph
from ._onnx_optimization_common import (
    _make_node, _rename_node_input,
    _make_att_graph, _rename_node_output
)


def _remove_node_identity_node(node, recursive, debug_info):
    """
    Removes *Identity* in subgraph.

    @param      node        onnx node
    @param      recursive   does it in subgraphs as well
    @return                 new node
    """
    if not hasattr(node, 'attribute'):
        return node
    modified = 0
    new_atts = []
    for att in node.attribute:
        if att.name == 'body':
            new_body = onnx_remove_node_identity(
                att.g, recursive=recursive,
                debug_info=debug_info + [att.name])
            new_atts.append(_make_att_graph(att.name, new_body))
            modified += 1
        else:
            new_atts.append(att)
    if modified > 0:
        new_node = _make_node(node.op_type, node.input,
                              node.output, name=node.name,
                              attributes=new_atts)
        return new_node
    return node


def onnx_remove_node_identity(onnx_model, recursive=True, debug_info=None):
    """
    Removes as many *Identity* nodes as possible.
    The function looks into every node and subgraphs if
    *recursive* is True for identity node. Unless such a
    node directy connects one input to one output, it will
    be removed and every other node gets its inputs or
    outputs accordingly renamed.

    @param      onnx_model      onnx model
    @param      recursive       looks into subgraphs
    @param      debug_info      debug information (private)
    @return                     new onnx _model
    """
    if debug_info is None:
        debug_info = [str(type(onnx_model)).split('.')[-1].strip("'>")]
    else:
        debug_info = debug_info + \
            [str(type(onnx_model)).split('.')[-1].strip("'>")]

    if hasattr(onnx_model, 'graph'):
        graph = onnx_remove_node_identity(
            onnx_model.graph, debug_info=debug_info + ['GRAPH'])
        new_model = make_model(graph)
        new_model.ir_version = onnx_model.ir_version
        new_model.producer_name = onnx_model.producer_name
        new_model.producer_version = onnx_model.producer_version
        new_model.domain = onnx_model.domain
        new_model.model_version = onnx_model.model_version
        new_model.doc_string = onnx_model.doc_string
        if hasattr(onnx_model, 'value_info'):
            graph.value_info.extend(onnx_model.value_info)
        return new_model

    graph = onnx_model

    inputs = set(i.name for i in graph.input)
    outputs = set(o.name for o in graph.output)

    def retrieve_idnodes(graph, existing_nodes):
        idnodes = []
        for i, (node, exnode) in enumerate(zip(graph.node, existing_nodes)):
            if exnode is None:
                continue
            if node.op_type == 'Identity':
                input = node.input[0]
                output = node.output[0]
                idnodes.append((i, node, input, output))
        return idnodes

    nodes = list(graph.node)
    rem = 1
    while rem > 0:
        rem = 0
        idnodes = retrieve_idnodes(graph, nodes)
        restart = False
        for i, _, inp, out in idnodes:
            if restart:
                break
            if nodes[i] is None:
                # Already removed.
                continue
            if inp in inputs and out in outputs:
                # Cannot be removed.
                continue
            if not restart and out not in outputs:
                # We cannot change an output name.
                for j in range(len(nodes)):  # pylint: disable=C0200
                    if nodes[j] is None:
                        continue
                    if out in nodes[j].input:
                        nodes[j] = _rename_node_input(nodes[j], out, inp)
                        rem += 1
                        if nodes[j] == 'Identity':
                            restart = True
                nodes[i] = None
                rem += 1
            if not restart and inp not in inputs:
                # We cannot change an input name.
                for j in range(len(nodes)):  # pylint: disable=C0200
                    if nodes[j] is None:
                        continue
                    if inp in nodes[j].output:
                        nodes[j] = _rename_node_output(nodes[j], inp, out)
                        rem += 1
                        if nodes[j] == 'Identity':
                            restart = True
                nodes[i] = None
                rem += 1

    if recursive:
        # Handles subgraphs.
        for i in range(len(nodes)):  # pylint: disable=C0200
            node = nodes[i]
            if node is None or not (node.attribute):  # pylint: disable=C0325
                continue
            nodes[i] = _remove_node_identity_node(
                node, recursive=True, debug_info=debug_info + [node.name])

    # Finally create the new graph.
    nodes = list(filter(lambda n: n is not None, nodes))
    graph = make_graph(nodes, onnx_model.name,
                       onnx_model.input, onnx_model.output,
                       onnx_model.initializer)

    graph.value_info.extend(onnx_model.value_info)  # pylint: disable=E1101
    return graph
