"""
@file
@brief Implements a class able to compute the predictions
from on an :epkg:`ONNX` model.
"""
from onnx import helper, shape_inference
from .onnx2py_helper import guess_proto_dtype


def enumerate_model_node_outputs(model, add_node=False, order=False):
    """
    Enumerates all the nodes of a model.

    :param model: :epkg:`ONNX` graph
    :param add_node: if False, the function enumerates
        all output names from every node, otherwise, it
        enumerates tuple (output name, node)
    :param order: goes through outputs following the graph order
    :return: enumerator
    """
    if not hasattr(model, "graph"):
        raise TypeError(  # pragma: no cover
            "Parameter model is not an ONNX model but "
            "{}".format(type(model)))
    if order:
        edges = []
        order = {}
        node_names = {}
        for inp in model.graph.input:
            order[0, inp.name] = 0
        for node in model.graph.node:
            order[1, node.name] = 0
            for i in node.input:
                edges.append(('in', i, node.name))
            for o in node.output:
                edges.append(('out', o, node.name))
                node_names[o] = node
                order[0, o] = 0

        modif = 1
        while modif > 0:
            modif = 0
            for kind, data_name, node_name in edges:
                if kind == 'in':
                    if (0, data_name) not in order:
                        continue
                    if order[0, data_name] + 1 > order[1, node_name]:
                        modif += 1
                        order[1, node_name] = order[0, data_name] + 1
                else:
                    if order[1, node_name] + 1 > order[0, data_name]:
                        modif += 1
                        order[0, data_name] = order[1, node_name] + 1

        orders = [(v, k) for k, v in order.items()]
        orders.sort()

        for _, k in orders:
            if k[0] == 1:
                continue
            out = k[1]
            if out not in node_names:
                continue
            yield (out, node_names[out]) if add_node else out
    else:
        for node in model.graph.node:
            for out in node.output:
                yield (out, node) if add_node else out


def select_model_inputs_outputs(model, outputs=None, inputs=None,
                                infer_shapes=False, overwrite=None,
                                verbose=0, fLOG=None):
    """
    Takes a model and changes its outputs.

    :param model: :epkg:`ONNX` model
    :param inputs: new inputs, same ones if None
    :param outputs: new outputs, same ones if None
    :param infer_shapes: infer inputs and outputs shapes
    :param overwrite: overwrite type and shapes for
        inputs or outputs, *overwrite* is a
        dictionary `{'name': (numpy dtype, shape)}`
    :param verbose: display information while converting
    :param fLOG: logging function
    :return: modified model

    The function removes unneeded nodes.

    .. exref::
        :title: Change ONNX model inputs

        The following exampels shows how to change the inputs of model
        to bypass the first nodes. Shape inferences fails to determine
        the new inputs type. They need to be overwritten.
        `verbose=1, fLOG=print` shows the number of deleted nodes.

        ::

            import onnx
            from mlprodict.onnx_tools.onnx_manipulations import select_model_inputs_outputs

            onx = onnx.load(path)
            onx2 = select_model_inputs_outputs(
                onx, inputs=["SentenceTokenizer/SentencepieceTokenizeOp:0",
                             "SentenceTokenizer/SentencepieceTokenizeOp:1"],
                infer_shapes=True, verbose=1, fLOG=print,
                overwrite={'SentenceTokenizer/SentencepieceTokenizeOp:0': (numpy.int32, None),
                           'SentenceTokenizer/SentencepieceTokenizeOp:1': (numpy.int64, None)})
            onnx.save(onx2, path2)

    .. versionchanged:: 0.6
        Supports the case where inputs are changed.
    """
    if inputs is not None and not isinstance(inputs, list):
        inputs = [inputs]
    if outputs is not None and not isinstance(outputs, list):
        outputs = [outputs]
    if inputs is None:
        inputs = [i.name for i in model.graph.input]
    if outputs is None:
        outputs = [o.name for o in model.graph.output]

    mark_var = {}
    for out in enumerate_model_node_outputs(model):
        mark_var[out] = 0
    for inp in inputs:
        mark_var[inp] = 0
    for out in outputs:
        if out not in mark_var:
            raise ValueError(  # pragma: no cover
                "Output '{}' not found in model.".format(out))
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
                if inp in inputs:
                    continue
                if mark_var.get(inp, 0) == 1:
                    continue
                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes if mark_op[node.name] == 1]

    known_shapes = {}
    if infer_shapes:
        shapes = shape_inference.infer_shapes(model)
        for shape in shapes.graph.value_info:  # pylint: disable=E1101
            known_shapes[shape.name] = shape.type
        for shape in shapes.graph.input:  # pylint: disable=E1101
            known_shapes[shape.name] = shape.type
        for shape in shapes.graph.output:  # pylint: disable=E1101
            known_shapes[shape.name] = shape.type
    else:
        for shape in model.graph.input:  # pylint: disable=E1101
            known_shapes[shape.name] = shape.type
        for shape in model.graph.output:  # pylint: disable=E1101
            known_shapes[shape.name] = shape.type

    var_in = []
    for name in inputs:
        if overwrite is not None and name in overwrite:
            dtype, shape = overwrite[name]
            proto_dtype = guess_proto_dtype(dtype)
            value_info = helper.make_tensor_value_info(
                name, proto_dtype, shape)
        elif name in known_shapes:
            info = known_shapes[name].tensor_type
            proto_dtype = info.elem_type
            if proto_dtype == 0:
                value_info = helper.ValueInfoProto()
                value_info.name = name
            else:
                shape = [getattr(d, 'dim_value', None) for d in info.shape.dim]
                shape = [None if s == 0 else s for s in shape]
                value_info = helper.make_tensor_value_info(
                    name, proto_dtype, shape)
        else:
            value_info = helper.ValueInfoProto()
            value_info.name = name
        var_in.append(value_info)

    var_out = []
    for name in outputs:
        if overwrite is not None and name in overwrite:
            dtype, shape = overwrite[name]
            proto_dtype = guess_proto_dtype(dtype)
            value_info = helper.make_tensor_value_info(
                name, proto_dtype, shape)
        elif name in known_shapes:
            info = known_shapes[name].tensor_type
            proto_dtype = info.elem_type
            if proto_dtype == 0:
                value_info = helper.ValueInfoProto()
                value_info.name = name
            else:
                shape = [getattr(d, 'dim_value', None) for d in info.shape.dim]
                shape = [None if s == 0 else s for s in shape]
                value_info = helper.make_tensor_value_info(
                    name, proto_dtype, shape)
        else:
            value_info = helper.ValueInfoProto()
            value_info.name = name
        var_out.append(value_info)

    if verbose > 0 and fLOG is not None:  # pragma: no cover
        fLOG("[select_model_inputs_outputs] nodes %r --> %r" % (
            len(model.graph.node), len(keep_nodes)))
        fLOG("[select_model_inputs_outputs] inputs: %r" % var_in)
        fLOG("[select_model_inputs_outputs] inputs: %r" % var_out)

    graph = helper.make_graph(keep_nodes, model.graph.name, var_in,
                              var_out, model.graph.initializer)
    onnx_model = helper.make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:  # pragma: no cover
        values = {p.key: p.value for p in model.metadata_props}
        helper.set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    return onnx_model
