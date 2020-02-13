"""
@file
@brief Implements a class able to compute the predictions
from on an :epkg:`ONNX` model.
"""
from onnx import helper


def enumerate_model_node_outputs(model, add_node=False):
    """
    Enumerates all the nodes of a model.

    @param      model       :epkg:`ONNX` graph
    @param      add_node    if False, the function enumerates
                            all output names from every node, otherwise, it
                            enumerates tuple (output name, node)
    @return                 enumerator
    """
    if not hasattr(model, "graph"):
        raise TypeError("Parameter model is not an ONNX model but "
                        "{}".format(type(model)))
    for node in model.graph.node:
        for out in node.output:
            yield (out, node) if add_node else out


def select_model_inputs_outputs(model, outputs=None, inputs=None):
    """
    Takes a model and changes its outputs.

    @param      model       :epkg:`ONNX` model
    @param      inputs      new inputs, same ones if None
    @param      outputs     new outputs, same ones if None
    @return                 modified model

    The function removes unneeded files.
    """
    if inputs is not None:
        raise NotImplementedError("Parameter inputs cannot be empty.")
    if outputs is None:
        raise RuntimeError("Parameter outputs cannot be None.")
    if not isinstance(outputs, list):
        outputs = [outputs]

    mark_var = {}
    for out in enumerate_model_node_outputs(model):
        mark_var[out] = 0
    for inp in model.graph.input:
        mark_var[inp.name] = 0
    for out in outputs:
        if out not in mark_var:
            raise ValueError("Output '{}' not found in model.".format(out))
        mark_var[out] = 1

    nodes = model.graph.node[::-1]
    mark_op = {}
    for node in nodes:
        mark_op[node.name] = 0

    # We mark all the nodes we need to keep.
    nb = 1
    while nb > 0:
        nb = 0
        for node in nodes:
            if mark_op[node.name] == 1:
                continue
            mod = False
            for out in node.output:
                if mark_var[out] == 1:
                    mark_op[node.name] = 1
                    mod = True
                    break
            if not mod:
                continue

            nb += 1
            for inp in node.input:
                if mark_var.get(inp, 0) == 1:
                    continue
                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes if mark_op[node.name] == 1]

    var_out = []
    for out in outputs:
        value_info = helper.ValueInfoProto()
        value_info.name = out
        var_out.append(value_info)
    graph = helper.make_graph(keep_nodes, model.graph.name, model.graph.input,
                              var_out, model.graph.initializer)
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

    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    if len(onnx_model.graph.input) != len(model.graph.input):  # pylint: disable=E1101
        raise RuntimeError("Input mismatch {} != {}".format(
            len(onnx_model.input), len(model.input)))  # pylint: disable=E1101
    return onnx_model
