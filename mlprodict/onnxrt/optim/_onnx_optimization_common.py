"""
@file
@brief Common functions to reduce the number of
nodes of an :epkg:`ONNX` graphs.
"""
from onnx.helper import make_graph, ValueInfoProto
from onnx import AttributeProto, NodeProto
from onnx.helper import make_attribute


def _make_node(op_type, inputs, outputs, name=None, doc_string=None,
               domain=None, attributes=None):
    """
    Constructs a NodeProto.

    :param op_type: (string): The name of the operator to construct
    :param inputs: list of input names
    :param outputs: list of output names
    :param name: optional unique identifier for NodeProto
    :param doc_string: optional documentation
        string for NodeProto
    :param domain: optional domain for NodeProto.
        If it's None, we will just use default domain (which is empty)
    :param attributes: the attributes of the node.  The acceptable values
        are documented in :func:`make_attribute`.
    :return: node
    """
    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)  # pylint: disable=E1101
    node.output.extend(outputs)  # pylint: disable=E1101
    if name:
        node.name = name
    if doc_string:
        node.doc_string = doc_string
    if domain is not None:
        node.domain = domain
    if isinstance(attributes, dict):
        if len(attributes) > 0:
            node.attribute.extend(  # pylint: disable=E1101
                make_attribute(key, value)
                for key, value in sorted(attributes.items()))
    elif attributes:
        for att in attributes:
            node.attribute.extend([att])  # pylint: disable=E1101
    return node


def _replace(name, old_name, new_name):
    if name == old_name:
        return new_name
    return name


def _rename_node_input(onnx_node, old_name, new_name):
    """
    Renames an input from a node.

    @param      onnx_node       onnx_node
    @param      old_name        old name
    @param      new_name        new name
    @return                     new node
    """
    inputs = [_replace(name, old_name, new_name) for name in onnx_node.input]
    outputs = list(onnx_node.output)
    if hasattr(onnx_node, 'attribute'):
        new_atts = []
        for att in onnx_node.attribute:
            if att.name == 'body':
                new_body = _rename_graph_input(att.g, old_name, new_name)
                attr = AttributeProto()
                attr.name = att.name
                attr.g.CopyFrom(new_body)  # pylint: disable=E1101
                attr.type = AttributeProto.GRAPH  # pylint: disable=E1101
                new_atts.append(attr)
            else:
                new_atts.append(att)
        atts = new_atts
    else:
        atts = onnx_node.attribute
    node = _make_node(onnx_node.op_type, inputs,
                      outputs, name=onnx_node.name,
                      attributes=atts)
    return node


def _rename_graph_output(graph, old_name, new_name):
    """
    Renames an output and adds an *Identity* node
    to connect the dots.

    @param      graph       ONNX graph
    @return                 modified graph
    """
    outputs = []
    for o in graph.output:
        if old_name != o.name:
            outputs.append(o)
        else:
            value_info = ValueInfoProto()
            value_info.name = new_name
            value_info.type.CopyFrom(o.type)  # pylint: disable=E1101
            if o.type.doc_string:
                value_info.doc_string = o.type.doc_string
            outputs.append(value_info)
    nodes = list(graph.node)
    nodes.append(_make_node('Identity', [old_name], [new_name]))
    new_graph = make_graph(nodes, graph.name, graph.input, outputs,
                           graph.initializer)
    new_graph.value_info.extend(graph.value_info)  # pylint: disable=E1101
    return new_graph


def _rename_graph_input(graph, old_name, new_name):
    """
    Renames an input and adds an *Identity* node
    to connect the dots.

    @param      graph       ONNX graph
    @return                 modified graph
    """
    inputs = []
    for i in graph.input:
        if old_name != i.name:
            inputs.append(i)
        else:
            value_info = ValueInfoProto()
            value_info.name = new_name
            value_info.type.CopyFrom(i.type)  # pylint: disable=E1101
            if i.type.doc_string:
                value_info.doc_string = i.type.doc_string
            inputs.append(value_info)
    nodes = list(graph.node)
    nodes.append(_make_node('Identity', [new_name], [old_name]))
    new_graph = make_graph(nodes, graph.name, inputs, graph.output,
                           graph.initializer)
    new_graph.value_info.extend(graph.value_info)  # pylint: disable=E1101
    return new_graph


def _make_att_graph(name, new_body):
    attr = AttributeProto()
    attr.name = name
    attr.g.CopyFrom(new_body)  # pylint: disable=E1101
    attr.type = AttributeProto.GRAPH  # pylint: disable=E1101
    return attr


def _rename_node_output(onnx_node, old_name, new_name):
    """
    Renames an output from a node.

    @param      onnx_node       onnx_node
    @param      old_name        old name
    @param      new_name        new name
    @return                     new node
    """
    inputs = list(onnx_node.input)
    outputs = [_replace(name, old_name, new_name) for name in onnx_node.output]
    if hasattr(onnx_node, 'attribute'):
        new_atts = []
        for att in onnx_node.attribute:
            if att.name == 'body':
                new_body = _rename_graph_output(att.g, old_name, new_name)
                new_atts.append(_make_att_graph(att.name, new_body))
            else:
                new_atts.append(att)
        atts = new_atts
    else:
        atts = onnx_node.attribute
    node = _make_node(onnx_node.op_type, inputs,
                      outputs, name=onnx_node.name,
                      attributes=atts)
    return node
