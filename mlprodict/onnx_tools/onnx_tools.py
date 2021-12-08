"""
@file
@brief Functions to manipulate ONNX file.
"""
from onnx import helper


def find_node_name(model, name):
    """
    Finds a node by its name.
    :param model: onnx graph
    :param name: node name
    :return: node pointer
    """
    if not hasattr(model, "graph"):
        raise TypeError(  # pragma: no cover
            "Parameter model is not an ONNX model but "
            "{}".format(type(model)))
    for node in model.graph.node:
        if node.name == name:
            return node
    return None  # pragma: no cover


def find_node_input_name(node, name):
    """
    Finds a node input by its name.
    :param node: onnx node
    :param name: node name
    :return: input index
    """
    for i, inode in enumerate(node.input.node):
        if inode.name == name:
            return i
    return -1


def insert_node(model, op_type, node, input_index=0, new_name=None, **attrs):
    """
    Inserts a node before one node input.
    :param model: onnx graph
    :param op_type:
    :param node: node or node name
    :param input_index: input index or input name
    :param attrs: node attributes
    :return: updated graph
    """
    if isinstance(node, str):
        inode = find_node_name(model, node)
    else:
        inode = node
    if isinstance(input_index, str):
        input_index_ = find_node_input_name(node, input_index)
        if input_index_ == -1:
            raise RuntimeError(  # pragma: no cover
                "Unable to find input_index %r in node %r." % (
                    input_index, node.name))  # pylint: disable=E1120
        input_index = input_index_

    # guess a new name
    names = []
    for n in model.graph.node:
        names.extend(n.input)
        names.extend(n.output)
    names = set(names)
    if new_name is None:
        new_name = op_type.lower()
    root_name = new_name
    i = 0
    while new_name in names:
        new_name = "%s_%d" % (root_name, i)
        i += 1

    new_node = helper.make_node(
        op_type, [inode.input[input_index]], [new_name], **attrs)
    inode.input[input_index] = new_name
    keep_nodes = list(model.graph.node)
    keep_nodes.append(new_node)
    keep_nodes = ensure_topological_order(
        model.graph.input, model.graph.initializer, keep_nodes)

    graph = helper.make_graph(
        keep_nodes, model.graph.name, model.graph.input,
        model.graph.output, model.graph.initializer)
    onnx_model = helper.make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        helper.set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    if len(onnx_model.graph.input) != len(model.graph.input):  # pylint: disable=E1101
        raise RuntimeError(  # pragma: no cover
            "Input mismatch {} != {}".format(
                len(onnx_model.input), len(model.input)))  # pylint: disable=E1101
    return onnx_model


def ensure_topological_order(inputs, initializers, nodes):
    """
    Ensures and modifies the order of nodes to have
    a topological order (every node in the list
    can only be an input for a node later in this list).
    The function raises an exception if a cycle is detected.

    :param inputs: graph inputs:
    :param initializers: graph initializers
    :param nodes: graph nodes
    :return: list ordered nodes
    """
    order = {}
    for inp in inputs:
        name = inp.name
        order[name] = 0
    for inp in initializers:
        name = inp.name
        order[name] = 0
    n_iter = 0
    while n_iter < len(nodes) * 2:
        n_iter += 1
        missing_names = set()
        missing_ops = []
        for node in nodes:
            maxi = 0
            for name in node.input:
                if name in order:
                    maxi = max(maxi, order[name])
                else:
                    maxi = None
                    missing_names.add(name)
                    break
            if maxi is None:
                missing_ops.append(node)
                continue
            key = id(node)
            if key in order:
                continue
            maxi += 1
            order[key] = maxi
            maxi += 1
            for name in node.output:
                if name in order:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to sort a node (cycle). An output was "
                        "already ordered %r (iteration=%r)." % (
                            name, n_iter))
                order[name] = maxi
        if len(missing_names) == 0:
            continue

    if len(missing_ops) > 0:  # pragma: no cover
        def nstr(name):
            if name in order:
                return "%s#%d" % (name, order[name])
            return name
        rows = ["%s(%s) -> [%s]" % (
            n.name or n.op_type,
            ', '.join(map(nstr, n.input)),
            ', '.join(n.output))
            for n in missing_ops]
        rows.insert(0, "")
        rows.append("--")
        rows.append("--all-nodes--")
        rows.append("--")
        rows.extend("%s(%s) -> [%s]" % (
            n.name or n.op_type,
            ', '.join(map(nstr, n.input)),
            ', '.join(n.output))
            for n in nodes)
        raise RuntimeError(
            "After %d iterations for %d nodes, still unable "
            "to sort names %r. The graph may be disconnected. "
            "List of operators: %s" % (
                n_iter, len(nodes), missing_names,
                "\n".join(rows)))

    # Update order
    topo = [(order[id(node)], str(id(node))) for node in nodes]
    topo.sort()
    map_nodes = {str(id(node)): node for node in nodes}
    return [map_nodes[_[1]] for _ in topo]
