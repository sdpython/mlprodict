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
            raise RuntimeError(
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
