"""
@file
@brief Functions to manipulate ONNX file.
"""
from collections import OrderedDict
from onnx import helper, TensorProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE


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


def reorder_nodes_for_display(nodes):
    """
    Reorders the node with breadth first seach (BFS).

    :param nodes: list of ONNX nodes
    :return: reordered list of nodes
    """
    all_outputs = set()
    all_inputs = set()
    for node in nodes:
        all_outputs |= set(node.output)
        all_inputs |= set(node.input)
    common = all_outputs & all_inputs
    dnodes = OrderedDict()
    successors = {}
    predecessors = {}
    for node in nodes:
        node_name = node.name + "".join(node.output)
        dnodes[node_name] = node
        successors[node_name] = set()
        predecessors[node_name] = set()
        for name in node.input:
            predecessors[node_name].add(name)
            if name not in successors:
                successors[name] = set()
            successors[name].add(node_name)
        for name in node.output:
            successors[node_name].add(name)
            predecessors[name] = {node_name}

    known = all_inputs - common
    new_nodes = []
    done = set()

    def _find_sequence(node_name, known, done):
        res = [node_name]
        while res[-1] in successors:
            next_names = successors[res[-1]]
            if res[-1] not in dnodes:
                next_names = set(v for v in next_names if v not in known)
                if len(next_names) == 1:
                    res.extend(next_names)
                elif len(next_names) == 0:
                    res.extend(next_names)
                    break
                else:
                    break
            else:
                next_names = set(v for v in next_names if v not in done)
                if len(next_names) == 1:
                    res.extend(next_names)
                elif len(next_names) == 0:
                    res.extend(next_names)
                    break
                else:
                    break

        return [r for r in res if r in dnodes]

    while len(done) < len(nodes):
        # possible
        possibles = OrderedDict()
        for k, v in dnodes.items():
            if k in done:
                continue
            if predecessors[k] < known:
                possibles[k] = v

        sequences = OrderedDict()
        for k, v in possibles.items():
            if k in done:
                continue
            sequences[k] = _find_sequence(k, known, done)

        # find the best sequence
        best = None
        for k, v in sequences.items():
            if best is None or len(v) > len(sequences[best]):
                best = k
        if best is None:
            raise RuntimeError(
                "Wrong implementationt.")

        # process the sequence
        for k in sequences[best]:
            v = dnodes[k]
            new_nodes.append(v)
            done.add(k)
            known |= set(v.output)

    return new_nodes


def simple_onnx_str(model, verbose=False, break_row='---'):
    """
    Displays an ONNX graph into text.

    :param model: ONNX graph
    :param verbose: display debugging information
    :param break_row: string to insert in the model
    :return: str
    """
    def get_type(obj0):
        obj = obj0
        if hasattr(obj, 'data_type'):
            if (obj.data_type == TensorProto.FLOAT and
                    hasattr(obj, 'float_data')):
                return TENSOR_TYPE_TO_NP_TYPE[TensorProto.FLOAT]
            if (obj.data_type == TensorProto.DOUBLE and
                    hasattr(obj, 'double_data')):
                return TENSOR_TYPE_TO_NP_TYPE[TensorProto.DOUBLE]
            if (obj.data_type == TensorProto.INT64 and
                    hasattr(obj, 'int64_data')):
                return TENSOR_TYPE_TO_NP_TYPE[TensorProto.INT64]
            raise RuntimeError(
                "Unable to guess type from %r." % obj0)
        if hasattr(obj, 'type'):
            obj = obj.type
        if hasattr(obj, 'tensor_type'):
            obj = obj.tensor_type
        if hasattr(obj, 'elem_type'):
            return TENSOR_TYPE_TO_NP_TYPE[obj.elem_type]
        raise RuntimeError(
            "Unable to guess type from %r." % obj0)

    def get_shape(obj):
        obj0 = obj
        if hasattr(obj, 'data_type'):
            if (obj.data_type == TensorProto.FLOAT and
                    hasattr(obj, 'float_data')):
                return (len(obj.float_data), )
            if (obj.data_type == TensorProto.DOUBLE and
                    hasattr(obj, 'double_data')):
                return (len(obj.double_data), )
            if (obj.data_type == TensorProto.INT64 and
                    hasattr(obj, 'int64_data')):
                return (len(obj.int64_data), )
            raise RuntimeError(
                "Unable to guess type from %r." % obj0)
        if hasattr(obj, 'type'):
            obj = obj.type
        if hasattr(obj, 'tensor_type'):
            obj = obj.tensor_type
        if hasattr(obj, 'shape'):
            obj = obj.shape
            dims = []
            for d in obj.dim:
                if hasattr(d, 'dim_value'):
                    dims.append(d.dim_value)
                else:
                    dims.append(None)
            return tuple(dims)
        raise RuntimeError(
            "Unable to guess type from %r." % obj0)

    def str_node(indent, node):
        return "%s%s(%s) -> %s" % (
            "  " * indent, node.op_type,
            ", ".join(node.input), ", ".join(node.output))

    rows = []
    if hasattr(model, 'opset_import'):
        for opset in model.opset_import:
            rows.append("opset: domain=%r version=%r" % (
                opset.domain, opset.version))
    if hasattr(model, 'graph'):
        model = model.graph

    # inputs
    for inp in model.input:
        rows.append("input: name=%r type=%r shape=%r" % (
            inp.name, get_type(inp), get_shape(inp)))
    # initializer
    for init in model.initializer:
        rows.append("init: name=%r type=%r shape=%r" % (
            init.name, get_type(init), get_shape(init)))

    # successors, predecessors
    successors = {}
    predecessors = {}
    for node in model.node:
        node_name = node.name + "".join(node.output)
        successors[node_name] = []
        predecessors[node_name] = []
        for name in node.input:
            predecessors[node_name].append(name)
            if name not in successors:
                successors[name] = []
            successors[name].append(node_name)
        for name in node.output:
            successors[node_name].append(name)
            predecessors[name] = [node_name]

    # walk through nodes
    init_names = set()
    indents = {}
    for inp in model.input:
        indents[inp.name] = 0
        init_names.add(inp.name)
    for init in model.initializer:
        indents[init.name] = 0
        init_names.add(init.name)

    nodes = reorder_nodes_for_display(model.node)

    previous_indent = None
    previous_out = None
    previous_in = None
    for node in nodes:
        add_break = False
        name = node.name + "".join(node.output)
        if name in indents:
            indent = indents[name]
            if previous_indent is not None and indent < previous_indent:
                if verbose:
                    print("[simple_onnx_str] break1 %s" % node.op_type)
                add_break = True
        elif previous_in is not None and set(node.input) == previous_in:
            indent = previous_indent
        else:
            inds = [indents[i] for i in node.input if i not in init_names]
            if len(inds) == 0:
                indent = 0
            else:
                mi, ma = min(inds), max(inds)
                indent = mi
                if previous_indent is not None and indent < previous_indent:
                    if verbose:
                        print("[simple_onnx_str] break2 %s" % node.op_type)
                    add_break = True
            if (not add_break and previous_out is not None and
                    len(set(node.input) & previous_out) == 0):
                if verbose:
                    print("[simple_onnx_str] break3 %s" % node.op_type)
                add_break = True
                indent = 0

        if add_break:
            rows.append(break_row)
        rows.append(str_node(indent, node))
        indents[name] = indent

        successor = successors[name]
        predecessor = predecessors[name]
        add_indent = 1  # 0 if len(successor) == 1 else 1
        for i, o in enumerate(node.output):
            indents[o] = indent + add_indent

        for o in node.output:
            if o in indents:
                continue
            indents[o] = indents[name]
        previous_indent = indents[name]
        previous_out = set(node.output)
        previous_in = set(node.input)

    # outputs
    for out in model.output:
        rows.append("output: name=%r type=%r shape=%r" % (
            out.name, get_type(out), get_shape(out)))
    return "\n".join(rows)
