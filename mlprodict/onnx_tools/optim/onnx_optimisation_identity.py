"""
@file
@brief Optimisation of :epkg:`ONNX` graphs.
"""
import logging
from onnx import FunctionProto, AttributeProto
from onnx.helper import make_graph, make_function
from ._onnx_optimisation_common import (  # pylint: disable=E0611
    _rename_node_input,
    _rename_node_output,
    _apply_optimisation_on_graph,
    _apply_remove_node_fct_node)


logger = logging.getLogger('onnx:optim')


def onnx_remove_node_identity(onnx_model, recursive=True, debug_info=None, **options):
    """
    Removes as many *Identity* nodes as possible.
    The function looks into every node and subgraphs if
    *recursive* is True for identity node. Unless such a
    node directy connects one input to one output, it will
    be removed and every other node gets its inputs or
    outputs accordingly renamed.

    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :param debug_info: debug information (private)
    :param options: additional options (unused)
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
            onnx_remove_node_identity, onnx_model,
            recursive=recursive, debug_info=debug_info, **options)

    graph = onnx_model
    logger.debug("onnx_remove_node_identity:begin with %d nodes.",
                 len(graph.node))
    is_function = isinstance(graph, FunctionProto)

    if is_function:
        inputs = set(graph.input)
        outputs = set(graph.output)
    else:
        inputs = set(i.name for i in graph.input)
        inits = set(i.name for i in graph.initializer)
        inputs_inits = inputs.union(inits)
        outputs = set(o.name for o in graph.output)

    def retrieve_idnodes(graph, existing_nodes):
        idnodes = []
        for i, exnode in enumerate(existing_nodes):
            if exnode is None:
                continue
            if exnode.op_type == 'Identity':
                input = exnode.input[0]
                output = exnode.output[0]
                idnodes.append((i, exnode, input, output))
        return idnodes

    # add to output the list of local variables in subgraphs
    def append_local_variable(graph, known=None, subgraph=True):
        if known is None:
            known = set()
        else:
            known = known.copy()
        local_var = set()
        if isinstance(graph, FunctionProto):
            known = set(graph.input)
        else:
            known = set(i.name for i in graph.input)
            known |= set(i.name for i in graph.initializer)
        for node in graph.node:
            for i in node.input:
                if i not in known and subgraph:
                    local_var.add(i)
            for o in node.output:
                known.add(o)
            for att in node.attribute:
                if (att.type == AttributeProto.GRAPH and  # pylint: disable=E1101
                        hasattr(att, 'g') and att.g is not None):
                    lv = append_local_variable(att.g, known)
                    local_var |= lv
        return local_var

    local_vars = append_local_variable(graph, subgraph=False)
    logger.debug('onnx_remove_node_identity:local_vars:%r', local_vars)
    ext_outputs = outputs | local_vars

    nodes = list(graph.node)
    rem = 1
    while rem > 0:
        rem = 0
        idnodes = retrieve_idnodes(graph, nodes)
        restart = False
        for i, _, inp, out in idnodes:
            if restart:
                break  # pragma: no cover
            if nodes[i] is None:
                # Already removed.
                continue  # pragma: no cover
            if inp in inputs_inits and out in ext_outputs:
                # Cannot be removed.
                continue
            if not restart and out not in ext_outputs:
                # We cannot change an output name.
                for j in range(len(nodes)):  # pylint: disable=C0200
                    if nodes[j] is None:
                        continue
                    if out in nodes[j].input:
                        logger.debug('onnx_remove_node_identity:'
                                     '_rename_node_input:%s:%r->%r:'
                                     'out=%r:inp=%r',
                                     nodes[j].op_type, nodes[j].input,
                                     nodes[j].output, out, inp)
                        nodes[j] = _rename_node_input(nodes[j], out, inp)
                        rem += 1
                        if nodes[j].op_type == 'Identity':
                            restart = True  # pragma: no cover
                logger.debug('onnx_remove_node_identity:1:remove:%s:%r->%r:',
                             nodes[i].op_type, nodes[i].input, nodes[i].output)
                nodes[i] = None
                rem += 1
                continue
            if not restart and inp not in inputs_inits and inp not in ext_outputs:
                # We cannot change an input name or an output name.
                for j in range(len(nodes)):  # pylint: disable=C0200
                    if nodes[j] is None:
                        continue
                    if inp in nodes[j].output:
                        logger.debug('onnx_remove_node_identity:'
                                     '_rename_node_output:%s:%r->%r:'
                                     'inp=%r:out=%r',
                                     nodes[j].op_type, nodes[j].input,
                                     nodes[j].output, inp, out)
                        nodes[j] = _rename_node_output(nodes[j], inp, out)
                        rem += 1
                        if nodes[j].op_type == 'Identity':
                            restart = True  # pragma: no cover
                    if inp in nodes[j].input:
                        logger.debug('onnx_remove_node_identity:'
                                     '_rename_node_input:%s:%r->%r:'
                                     'inp=%r:out=%r',
                                     nodes[j].op_type, nodes[j].input,
                                     nodes[j].output, inp, out)
                        nodes[j] = _rename_node_input(nodes[j], inp, out)
                        rem += 1
                        if nodes[j].op_type == 'Identity':
                            restart = True
                logger.debug('onnx_remove_node_identity:2:remove:%s:%r->%r:',
                             nodes[i].op_type, nodes[i].input, nodes[i].output)
                nodes[i] = None
                rem += 1

    if recursive:
        # Handles subgraphs.
        for i in range(len(nodes)):  # pylint: disable=C0200
            node = nodes[i]
            if node is None or not (node.attribute):  # pylint: disable=C0325
                continue
            nodes[i] = _apply_remove_node_fct_node(
                onnx_remove_node_identity,
                node, recursive=True, debug_info=debug_info + [node.name])

    # Finally create the new graph.
    nodes = list(filter(lambda n: n is not None, nodes))
    if len(nodes) == 0:
        # something went wrong
        nodes = graph.node
    if is_function:
        logger.debug("onnx_remove_node_identity:end function with %d nodes.",
                     len(nodes))
        return make_function(
            onnx_model.domain, onnx_model.name,
            onnx_model.input, onnx_model.output, nodes,
            opset_imports=onnx_model.opset_import,
            attributes=onnx_model.attribute,
            doc_string=onnx_model.doc_string)

    graph = make_graph(nodes, onnx_model.name,
                       onnx_model.input, onnx_model.output,
                       onnx_model.initializer)

    graph.value_info.extend(onnx_model.value_info)  # pylint: disable=E1101
    logger.debug("onnx_remove_node_identity: end graph with %d nodes.",
                 len(nodes))
    return graph
