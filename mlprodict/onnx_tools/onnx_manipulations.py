"""
@file
@brief Implements a class able to compute the predictions
from on an :epkg:`ONNX` model.
"""
import hashlib
from onnx import helper, shape_inference
from .onnx2py_helper import guess_proto_dtype, from_array
from .optim import onnx_remove_node_unused


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
                                remove_unused=True,
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
    :param remove_unused: remove unused nodes from the graph
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

    .. versionchanged:: 0.7
        Parameter *remove_unused* was added. Unused are removed by default.
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
                if len(shape) == 0:
                    shape = None
                else:
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
                if len(shape) == 0:
                    shape = None
                else:
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

    # remove unused nodes
    if remove_unused:
        onnx_model = onnx_remove_node_unused(onnx_model, recursive=False)

    return onnx_model


def overwrite_opset(model, new_opset):
    """
    Overwrites the main opset in an ONNX file.
    Does not change any node definition.

    :param model: ONNX model
    :param new_opset: new opset
    :return: ONNX model
    """
    graph = helper.make_graph(
        model.graph.node, model.graph.name, model.graph.input,
        model.graph.output, model.graph.initializer)
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
        if oimp.domain == '':
            op_set.domain = oimp.domain
            op_set.version = new_opset
        else:
            op_set.domain = oimp.domain
            op_set.version = oimp.version
    return onnx_model


def hash_onnx_object(obj, max_size):
    """
    Hash the content of an object.
    """
    m = hashlib.sha256()
    if hasattr(obj, 'op_type'):
        # An operator.
        m.update(obj.op_type.encode('ascii'))
        m.update(str(len(obj.input)).encode('ascii'))
        m.update(str(len(obj.output)).encode('ascii'))
        if hasattr(obj, 'attribute'):
            for att in obj.attribute:
                m.update(att.name.encode('ascii'))
                m.update(att.SerializeToString())
    else:
        # An initializer.
        name = obj.name
        docf = obj.doc_string
        obj.name = ''
        obj.doc_string = ''
        try:
            m.update(obj.SerializeToString())
        except AttributeError as e:  # pragma: no cover
            raise RuntimeError(
                "Unable to hash object type %r, value=%r."
                "" % (type(obj), obj)) from e
        finally:
            obj.name = name
            obj.doc_string = docf

    content = m.hexdigest()
    if len(content) > max_size:
        content = content[:max_size]
    return content.upper()


def onnx_rename_names(model, strategy='simple', recursive=True,
                      verbose=0, fLOG=print,
                      counts=None, replace=None, taken=None):
    """
    Renames all names except the inputs and outputs.

    :param model: onnx model
    :param strategy: two strategies are implemented, see below
    :param recursive: walk through subgraphs
    :param verbose: verbose, if positive, reports on all changed names
    :param fLOG: logging function
    :param counts: used for recursion
    :param replace: used for recursion, it can be also used to
        to fix some replacements
    :param taken: used for recursion
    :return: onnx model (the model is modified in place)

    Strategies:
    * `'simple'`: use a letter `n` for node, `r`, `i` for initializer,
        this letter is followed by a number
    * `'type'`: the name depends on the node type and content,
        the hash is kept as small as possible
    """
    counts = counts or {'init': 0, 'node': 0, 'result': 0}
    replace = replace or {}
    taken = taken or set()
    graph = model.graph if hasattr(model, 'graph') else model

    for obj in graph.input:
        replace[obj.name] = obj.name
    for obj in graph.output:
        replace[obj.name] = obj.name

    def _check_name_simple(prefix):
        if prefix not in replace:
            return prefix
        c = 1
        final = "%s_%d" % (prefix, c)
        while final in taken:
            c += 1
            final = "%s_%d" % (prefix, c)
        taken.add(final)
        return final

    def _check_name_type(obj, prefix):
        c = 2
        hash = hash_onnx_object(obj, c)
        final = "%s_%s" % (prefix, hash)
        while final in taken:
            c += 2
            hash = hash_onnx_object(obj, c)
            final = "%s_%s" % (prefix, hash)
        taken.add(final)
        return final

    def get_name_init(init):
        if init.name in replace:
            return replace[init.name]
        if strategy == 'simple':
            name = _check_name_simple('i%d' % counts['init'])
            counts['init'] += 1
            replace[init.name] = name
            if verbose > 0 and fLOG is not None:
                fLOG('[onnx_rename_names] %r -> %r' % (init.name, name))
            return name
        if strategy == 'type':
            name = _check_name_type(init, 'i')
            counts['init'] += 1
            replace[init.name] = name
            if verbose > 0 and fLOG is not None:
                fLOG('[onnx_rename_names] %r -> %r' % (init.name, name))
            return name
        raise ValueError(  # pragma: no cover
            "Unknown strategy %r." % strategy)

    def get_name_node(node):
        if node.name in replace:
            return replace[node.name]
        if strategy == 'simple':
            name = _check_name_simple('n%d' % counts['node'])
            counts['node'] += 1
            replace[node.name] = name
            if verbose > 0 and fLOG is not None:
                fLOG('[onnx_rename_names] %r -> %r' % (node.name, name))
            return name
        if strategy == 'type':
            name = _check_name_type(node, 'n')
            counts['node'] += 1
            replace[node.name] = name
            if verbose > 0 and fLOG is not None:
                fLOG('[onnx_rename_names] %r -> %r' % (node.name, name))
            return name
        raise ValueError(  # pragma: no cover
            "Unknown strategy %r." % strategy)

    def get_name_result(node, i, name, suffix):
        if name in replace:
            return replace[name]
        if strategy == 'simple':
            new_name = _check_name_simple('r%d' % counts['result'])
            counts['result'] += 1
            replace[name] = new_name
            if verbose > 0 and fLOG is not None:
                fLOG('[onnx_rename_names] %r -> %r' % (name, new_name))
            return new_name
        if strategy == 'type':
            new_name = _check_name_type(node, 'r%s%d' % (suffix, i))
            counts['result'] += 1
            replace[name] = new_name
            if verbose > 0 and fLOG is not None:
                fLOG('[onnx_rename_names] %r -> %r' % (name, new_name))
            return new_name
        raise ValueError(  # pragma: no cover
            "Unknown strategy %r." % strategy)

    def get_name_input(node, i):
        return get_name_result(node, i, node.input[i], 'i')

    def get_name_output(node, i):
        return get_name_result(node, i, node.output[i], 'o')

    for init in graph.initializer:
        init.name = get_name_init(init)

    for node in graph.node:
        node.name = get_name_node(node)
        for i in range(len(node.input)):  # pylint: disable=C0200
            node.input[i] = get_name_input(node, i)
        for i in range(len(node.output)):  # pylint: disable=C0200
            node.output[i] = get_name_output(node, i)
        if not recursive or node.op_type not in {'Scan', 'If', 'Loop'}:
            continue
        # recursion
        for att in node.attribute:
            if att.name not in {'if_branch', 'else_branch', 'body'}:
                continue
            onnx_rename_names(
                att.g, strategy=strategy, fLOG=fLOG, verbose=verbose,
                counts=counts, replace=replace, taken=taken)

    return model


def insert_results_into_onnx(model, results, as_parameter=True, suffix='_DBG',
                             param_name=None, node_type='DEBUG',
                             domain='DEBUG', domain_opset=1):
    """
    Inserts results into an ONNX graph to produce an extended
    ONNX graph. It can saved and looked into with a tool such as
    :epkg:`netron`.

    :param model: ONNX graph
    :param results: results to be added in a dictionary
    :param as_parameter: add new nodes with results as one parameter
        (True) or as initializer (False)
    :param suffix: suffix to add to new results
    :param param_name: name of the parameter to add
        (by default the result name), it can be a function
        `param_name(reult_name) -> parameter_name`
    :param node_type: type of the new node
    :param domain: domain the new node
    :param domain_opset: opset for *domain*
    :return: new ONNX graph

    See method :meth:`OnnxInference.run2onnx
    <mlprodict.onnxrt.onnx_inference.OnnxInference.run2onnx>`
    to see a graph this function produces.

    .. image:: debug.png

    .. versionadded:: 0.7
    """
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    inits = list(model.graph.initializer)
    nodes = {id(n): n for n in model.graph.node}
    order = {id(n): i for i, n in enumerate(model.graph.node)}
    nodes_copy = {}

    names_init = set(init.name for init in inits)
    names_input = set(init.name for init in inputs)
    names_output = {}
    for node in nodes.values():
        for i, o in enumerate(node.output):
            names_output[o] = (i, node)

    for k, v in results.items():
        if k in names_init:
            # initializer are not inserted again
            continue
        if k in names_input:
            # inputs are added as
            raise NotImplementedError(
                "Unable to add debug information on input %r." % k)

        if k not in names_output:
            raise RuntimeError(
                "Unable to find result %r in the ONNX graph. Available="
                "[%s]." % (k, ", ".join(sorted(names_output))))

        index, node = names_output[k]
        new_name = k + suffix

        if id(node) not in nodes_copy:
            new_node = helper.make_node(
                node.op_type, list(node.input), list(node.output),
                domain=node.domain if node.domain else None,
                name=node.name + suffix)
            new_node.attribute.extend(node.attribute)  # pylint: disable=E1101
            nodes_copy[id(node)] = new_node
            order[id(new_node)] = order[id(node)]
        new_node = nodes_copy[id(node)]
        new_node.output[index] = new_name

        if as_parameter:
            pname = k if param_name is None else param_name(k)
            atts = {pname: from_array(v, name=pname)}
            inserted_node = helper.make_node(
                node_type, [new_name], [k], domain=domain,
                **atts)
        else:
            pname = k if param_name is None else param_name(k)
            pname += suffix + 'i'
            inserted_node = helper.make_node(
                node_type, [new_name, pname], [k], domain=domain)
            inits.append(from_array(v, name=pname))

        order[id(inserted_node)] = order[id(node)] + 1. / (index + 2)
        nodes[id(inserted_node)] = inserted_node

    new_nodes = [(order[id(n)], n)
                 for n in nodes.values() if id(n) not in nodes_copy]
    new_nodes.extend((order[id(n)], n) for n in nodes_copy.values())
    new_nodes = [n[1] for n in sorted(new_nodes)]

    graph = helper.make_graph(new_nodes, model.graph.name, inputs,
                              outputs, inits)
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
    op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
    op_set.domain = domain
    op_set.version = domain_opset
    return onnx_model
