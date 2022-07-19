# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=E1101, C0302

"""
@file
@brief Implements a class able to compute the predictions
from on an :epkg:`ONNX` model.
"""
import hashlib
from collections import Counter
import pprint
from onnx import (
    shape_inference, ModelProto, FunctionProto, GraphProto,
    AttributeProto)
from onnx.helper import (
    make_tensor_value_info, ValueInfoProto, set_model_props,
    make_graph, make_function, make_model, make_node,
    make_operatorsetid, make_attribute, make_value_info)
from .onnx2py_helper import (
    guess_proto_dtype, from_array, get_tensor_shape,
    get_tensor_elem_type)
from .optim import onnx_remove_node_unused
from .onnx_tools import enumerate_onnx_names, enumerate_onnx_nodes
from ..onnx_tools.onnx2py_helper import _var_as_dict, from_array


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
            f"Parameter model is not an ONNX model but {type(model)}")
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
        n_iter = 0
        while modif > 0 and n_iter <= len(model.graph.node):
            modif = 0
            n_iter += 1
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


def get_opsets(model, include_functions=True, exc=True):
    """
    Enumerates all opsets used in a model.

    :param model: :epkg:`ModelProto` or :epkg:`FunctionProto`
    :param include_functions: include opsets used in functions
    :param exc: raise an exception if conflicts are detected
    :return: dictionary
    """
    if isinstance(model, ModelProto):
        res = {}
        for op in model.opset_import:
            if exc and op.domain in res:
                raise ValueError(  # pragma: no cover
                    f"Domain {op.domain!r} appears multiple times.")
            res[op.domain] = op.version
        if include_functions:
            for f in model.functions:
                ops = get_opsets(f, exc=exc)
                for k, v in ops.items():
                    if k in res:
                        if res[k] != v:
                            if exc:
                                raise ValueError(  # pragma: no cover
                                    "Domain %r has different version in "
                                    "main graph (%d) and function %r "
                                    "(%d)." % (k, res[k], f.name, v))
                            res[k] = max(res[k], v)
                    else:
                        res[k] = v
        return res

    res = {}
    for op in model.opset_import:
        if exc and op.domain in res:
            raise ValueError(  # pragma: no cover
                f"Domain {op.domain!r} appears multiple times.")
        res[op.domain] = op.version
    return res


def get_hidden_inputs(nodes):
    """
    Returns the list of hidden inputs used by subgraphs.

    :param nodes: list of nodes
    :return: list of names
    """
    inputs = set()
    outputs = set()
    for node in nodes:
        inputs |= set(node.input)
        outputs |= set(node.output)
        for att in node.attribute:
            if (att.type != AttributeProto.GRAPH or  # pylint: disable=E1101
                    not hasattr(att, 'g') or att.g is None):
                continue
            hidden = get_hidden_inputs(att.g.node)
            inits = set(att.g.initializer)
            inputs |= hidden - (inits & hidden)
    return inputs - (outputs & inputs)


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
                f"Output '{out}' not found in model.")
        mark_var[out] = 1

    nodes = model.graph.node[::-1]
    mark_op = {}
    for node in nodes:
        mark_op[id(node)] = 0

    # We mark all the nodes we need to keep.
    nb = 1
    while nb > 0:
        nb = 0
        for node in nodes:
            if mark_op[id(node)] == 1:
                continue
            mod = False
            for out in node.output:
                if mark_var[out] == 1:
                    mark_op[id(node)] = 1
                    mod = True
                    break
            if not mod:
                continue

            hidden = get_hidden_inputs([node])
            node_inputs = list(node.input) + list(hidden)

            nb += 1
            for inp in node_inputs:
                if inp in inputs:
                    continue
                if mark_var.get(inp, 0) == 1:
                    continue
                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes[::-1] if mark_op[id(node)] == 1]

    if verbose > 1 and fLOG is not None:  # pragma: no cover
        for node in nodes:
            s = "+" if mark_op[id(node)] == 1 else "-"
            fLOG("[select_model_inputs_outputs] %s %s (%s) -> %s [%s]" % (
                s, node.op_type, ", ".join(node.input),
                ', '.join(node.output), node.name))

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
            value_info = make_tensor_value_info(
                name, proto_dtype, shape)
        elif name in known_shapes:
            info = known_shapes[name].tensor_type
            proto_dtype = info.elem_type
            if proto_dtype == 0:
                value_info = ValueInfoProto()
                value_info.name = name
            else:
                shape = get_tensor_shape(known_shapes[name])
                value_info = make_tensor_value_info(
                    name, proto_dtype, shape)
        else:
            value_info = ValueInfoProto()
            value_info.name = name
        var_in.append(value_info)

    var_out = []
    for name in outputs:
        if overwrite is not None and name in overwrite:
            dtype, shape = overwrite[name]
            proto_dtype = guess_proto_dtype(dtype)
            value_info = make_tensor_value_info(
                name, proto_dtype, shape)
        elif name in known_shapes:
            info = known_shapes[name].tensor_type
            proto_dtype = info.elem_type
            if proto_dtype == 0:
                value_info = ValueInfoProto()
                value_info.name = name
            else:
                shape = get_tensor_shape(known_shapes[name])
                value_info = make_tensor_value_info(
                    name, proto_dtype, shape)
        else:
            value_info = ValueInfoProto()
            value_info.name = name
        var_out.append(value_info)

    if verbose > 0 and fLOG is not None:  # pragma: no cover
        fLOG("[select_model_inputs_outputs] nodes %r --> %r" % (
            len(model.graph.node), len(keep_nodes)))
        fLOG("[select_model_inputs_outputs] inputs: %r" %
             [_.name for _ in var_in])
        fLOG("[select_model_inputs_outputs] inputs: %r" %
             [_.name for _ in var_out])

    graph = make_graph(keep_nodes, model.graph.name, var_in,
                       var_out, model.graph.initializer,
                       sparse_initializer=model.graph.sparse_initializer)
    onnx_model = make_model(graph, functions=model.functions)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:  # pragma: no cover
        values = {p.key: p.value for p in model.metadata_props}
        set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    # remove unused nodes
    if remove_unused:
        onnx_model = onnx_remove_node_unused(onnx_model, recursive=False)

    return onnx_model


def change_input_type(onx, changes):
    """
    Changes the type of an input.

    :param onx: ONNX model
    :param changes: dictionary '{ name: new proto element type }`
    :return: new onx
    """
    new_inputs = []
    for inp in onx.graph.input:
        if inp.name not in changes:
            new_inputs.append(inp)
            continue
        value_info = make_tensor_value_info(
            inp.name, changes[inp.name], None)
        new_inputs.append(value_info)

    # final
    graph = make_graph(list(onx.graph.node),
                       onx.graph.name, new_inputs,
                       list(onx.graph.output),
                       onx.graph.initializer,
                       sparse_initializer=onx.graph.sparse_initializer)
    onnx_model = make_model(graph, functions=onx.functions)
    onnx_model.ir_version = onx.ir_version
    onnx_model.producer_name = onx.producer_name
    onnx_model.producer_version = onx.producer_version
    onnx_model.domain = onx.domain
    onnx_model.model_version = onx.model_version
    onnx_model.doc_string = onx.doc_string
    if len(onx.metadata_props) > 0:  # pragma: no cover
        values = {p.key: p.value for p in onx.metadata_props}
        set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for oimp in onx.opset_import:
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    return onnx_model


def _change_subgraph_io_type_shape_list(io_list, type_changes, shape_changes):
    ms = False
    new_inputs = []
    for inp in io_list:
        m = False
        if isinstance(shape_changes, dict):
            if inp.name in shape_changes:
                shape = shape_changes[inp.name]
                m = True
            else:
                shape = get_tensor_shape(inp)
        else:
            shape = shape_changes(inp)
            m = True

        if isinstance(type_changes, dict):
            if inp.name in type_changes:
                ntype = type_changes[inp.name]
                m = True
            else:
                ntype = get_tensor_elem_type(inp)
        else:
            ntype = type_changes(inp)
            m = True

        if m:
            ms = True
            value_info = make_tensor_value_info(inp.name, ntype, shape)
            new_inputs.append(value_info)
        else:
            new_inputs.append(inp)
    return new_inputs if ms else None


def change_subgraph_io_type_shape(onx, type_changes=None, shape_changes=None,
                                  recursive=True):
    """
    Changes the type of an input or an output of a subgraph.

    :param onx: ModelProto, GraphProto
    :param type_changes: dictionary '{ name: new proto element type }`
        or function `f(input) -> type`
    :param shape_changes: dictionary '{ name: new shape }`
        or function `f(input) -> shape`
    :param recursive: True
    :return: new onx
    """
    if isinstance(onx, ModelProto):
        graph = change_subgraph_io_type_shape(
            onx.graph, type_changes, shape_changes, recursive)
        onnx_model = make_model(graph, functions=onx.functions)
        onnx_model.ir_version = onx.ir_version
        onnx_model.producer_name = onx.producer_name
        onnx_model.producer_version = onx.producer_version
        onnx_model.domain = onx.domain
        onnx_model.model_version = onx.model_version
        onnx_model.doc_string = onx.doc_string
        if len(onx.metadata_props) > 0:  # pragma: no cover
            values = {p.key: p.value for p in onx.metadata_props}
            set_model_props(onnx_model, values)

        del onnx_model.opset_import[:]  # pylint: disable=E1101
        for oimp in onx.opset_import:
            op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
            op_set.domain = oimp.domain
            op_set.version = oimp.version
        return onnx_model

    graph = onx
    new_inputs = _change_subgraph_io_type_shape_list(
        graph.input, type_changes or {}, shape_changes or {})
    if new_inputs is None:
        new_inputs = graph.input

    new_outputs = _change_subgraph_io_type_shape_list(
        graph.output, type_changes or {}, shape_changes or {})
    if new_outputs is None:
        new_outputs = graph.output

    # recursive
    if recursive:
        new_nodes = []
        for node in graph.node:
            modified = False
            atts = []
            for att in node.attribute:
                if (att.type == AttributeProto.GRAPH and
                        hasattr(att, 'g') and att.g is not None):
                    modified = True
                    g = change_subgraph_io_type_shape(
                        att.g, type_changes, shape_changes,
                        recursive=recursive)
                    att = make_attribute(att.name, g)
                atts.append(att)
            if modified:
                node = make_node(node.op_type, node.input, node.output)
                node.attribute.extend(atts)
            new_nodes.append(node)
    else:
        new_nodes = list(graph.node)

    # final
    graph = make_graph(new_nodes, graph.name, new_inputs, new_outputs,
                       graph.initializer,
                       sparse_initializer=graph.sparse_initializer)
    return graph


def overwrite_opset(model, new_opset):
    """
    Overwrites the main opset in an ONNX file.
    Does not change any node definition.

    :param model: ONNX model
    :param new_opset: new opset
    :return: ONNX model
    """
    graph = make_graph(
        model.graph.node, model.graph.name, model.graph.input,
        model.graph.output, model.graph.initializer,
        sparse_initializer=model.graph.sparse_initializer)
    onnx_model = make_model(graph, functions=model.functions)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:  # pragma: no cover
        values = {p.key: p.value for p in model.metadata_props}
        set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = new_opset if oimp.domain == '' else oimp.version
    return onnx_model


def hash_onnx_object(obj, max_size):
    """
    Hashes the content of an object.
    It uses module :mod:`hashlib`.

    :param obj: onnx graph (it must have a method `SerializeToString`)
    :param max_size: size of the hash
    :return: hash
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
                f"Unable to hash object type {type(obj)!r}, value={obj!r}.") from e
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
        final = f"{prefix}_{hash}"
        while final in taken:
            c += 2
            hash = hash_onnx_object(obj, c)
            final = f"{prefix}_{hash}"
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
                fLOG(f'[onnx_rename_names] init: {init.name!r} -> {name!r}')
            return name
        if strategy == 'type':
            name = _check_name_type(init, 'i')
            counts['init'] += 1
            replace[init.name] = name
            if verbose > 0 and fLOG is not None:
                fLOG(f'[onnx_rename_names] init: {init.name!r} -> {name!r}')
            return name
        raise ValueError(  # pragma: no cover
            f"Unknown strategy {strategy!r}.")

    def get_name_node(node):
        node_name = 'node_%s_%d' % (node.name, id(node))
        if node_name in replace:
            return replace[node_name]
        if strategy == 'simple':
            name = _check_name_simple('n%d' % counts['node'])
            counts['node'] += 1
            replace[node_name] = name
            if verbose > 0 and fLOG is not None:
                fLOG(f'[onnx_rename_names] node: {node_name!r} -> {name!r}')
            return name
        if strategy == 'type':
            name = _check_name_type(node, 'n')
            counts['node'] += 1
            replace[node_name] = name
            if verbose > 0 and fLOG is not None:
                fLOG(f'[onnx_rename_names] node: {node_name!r} -> {name!r}')
            return name
        raise ValueError(  # pragma: no cover
            f"Unknown strategy {strategy!r}.")

    def get_name_result(node, i, name, suffix):
        if name in replace:
            return replace[name]
        if strategy == 'simple':
            new_name = _check_name_simple('r%d' % counts['result'])
            counts['result'] += 1
            replace[name] = new_name
            if verbose > 0 and fLOG is not None:
                fLOG(f'[onnx_rename_names] result: {name!r} -> {new_name!r}')
            return new_name
        if strategy == 'type':
            new_name = _check_name_type(node, 'r%s%d' % (suffix, i))
            counts['result'] += 1
            replace[name] = new_name
            if verbose > 0 and fLOG is not None:
                fLOG(f'[onnx_rename_names] result: {name!r} -> {new_name!r}')
            return new_name
        raise ValueError(  # pragma: no cover
            f"Unknown strategy {strategy!r}.")

    def get_name_input(node, i):
        return get_name_result(node, i, node.input[i], 'i')

    def get_name_output(node, i):
        return get_name_result(node, i, node.output[i], 'o')

    for init in graph.initializer:
        init.name = get_name_init(init)
    for init in graph.sparse_initializer:
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


def onnx_rename_inputs_outputs(onx, rename):
    """
    Renames input or outputs names.

    :param onx: GraphProto, ModelProto
    :param rename: dictionary `{old_name: new_name}`
    :return: new onx
    """
    if isinstance(onx, ModelProto):
        graph = onnx_rename_inputs_outputs(onx.graph, rename)
        onnx_model = make_model(graph, functions=onx.functions)
        onnx_model.ir_version = onx.ir_version
        onnx_model.producer_name = onx.producer_name
        onnx_model.producer_version = onx.producer_version
        onnx_model.domain = onx.domain
        onnx_model.model_version = onx.model_version
        onnx_model.doc_string = onx.doc_string
        if len(onx.metadata_props) > 0:  # pragma: no cover
            values = {p.key: p.value for p in onx.metadata_props}
            set_model_props(onnx_model, values)

        del onnx_model.opset_import[:]  # pylint: disable=E1101
        for oimp in onx.opset_import:
            op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
            op_set.domain = oimp.domain
            op_set.version = oimp.version
        return onnx_model

    graph = onx
    new_inputs = []
    for inp in graph.input:
        if inp.name not in rename:
            new_inputs.append(inp)
            continue
        value_info = make_tensor_value_info(
            rename[inp.name], get_tensor_elem_type(inp), get_tensor_shape(inp))
        new_inputs.append(value_info)

    new_outputs = []
    for inp in graph.output:
        if inp.name not in rename:
            new_outputs.append(inp)
            continue
        value_info = make_tensor_value_info(
            rename[inp.name], get_tensor_elem_type(inp), get_tensor_shape(inp))
        new_outputs.append(value_info)

    new_inits = []
    for init in graph.initializer:
        if init.name in rename:
            init.name = rename[init.name]
        new_inits.append(init)

    new_sparse_inits = []
    for init in graph.sparse_initializer:
        if init.name in rename:
            init.name = rename[init.name]
        new_sparse_inits.append(init)

    new_nodes = []
    for node in graph.node:
        modified = False
        atts = []
        for att in node.attribute:
            if (att.type == AttributeProto.GRAPH and
                    hasattr(att, 'g') and att.g is not None):
                modified = True
                g = onnx_rename_inputs_outputs(att.g, rename)
                att = make_attribute(att.name, g)
            atts.append(att)
        if modified:
            node = make_node(node.op_type, node.input, node.output)
            node.attribute.extend(atts)

        inp = [rename.get(i, i) for i in node.input]
        out = [rename.get(i, i) for i in node.output]
        if inp == list(node.input) and out == list(node.output):
            new_nodes.append(node)
            continue

        node = make_node(node.op_type, inp, out, domain=node.domain,
                         name=node.name)
        node.attribute.extend(atts)
        new_nodes.append(node)

    # final
    graph = make_graph(new_nodes, graph.name, new_inputs, new_outputs,
                       new_inits, sparse_initializer=new_sparse_inits)
    return graph


def onnx_replace_functions(model, replace):
    """
    Replaces some of the function in model.

    :param model: *ModelProto*
    :param replace: dictionary `{ (domain, name): FunctionProto }`
    :return: new model
    """
    if not isinstance(model, ModelProto):
        raise TypeError(  # pragma: no cover
            f"Unexpected type {type(model)!r}.")
    new_functions = []
    modified = False
    for fct in model.functions:
        key = fct.domain, fct.name
        if key in replace:
            modified = True
            f = replace[key]
            if not isinstance(f, FunctionProto):
                raise TypeError(  # pragma: no cover
                    f"Unexpected type {type(f)!r} for function {key!r} in replace.")
            if len(f.input) != len(fct.input):
                raise ValueError(  # pragma: no cover
                    f"Input mismatches {f.input!r} != {fct.input!r} (expected).")
            if len(f.output) != len(fct.output):
                raise ValueError(  # pragma: no cover
                    f"Output mismatches {f.output!r} != {fct.output!r} (expected).")
            new_functions.append(f)
        else:
            new_functions.append(fct)
    if not modified:
        return model
    opsets = [make_operatorsetid(op.domain, op.version)
              for op in model.opset_import]
    onnx_model = make_model(
        model.graph, opset_imports=opsets, functions=new_functions)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:  # pragma: no cover
        values = {p.key: p.value for p in model.metadata_props}
        set_model_props(onnx_model, values)
    return onnx_model


def insert_results_into_onnx(model, results, as_parameter=True, suffix='_DBG',
                             param_name=None, node_type='DEBUG',
                             domain='DEBUG', domain_opset=1):
    """
    Inserts results into an ONNX graph to produce an extended
    ONNX graph. It can be saved and looked into with a tool such as
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
    inits_sparse = list(model.graph.sparse_initializer)
    nodes = {id(n): n for n in model.graph.node}
    order = {id(n): i for i, n in enumerate(model.graph.node)}
    nodes_copy = {}

    names_init = (set(init.name for init in inits) |
                  set(init.name for init in inits_sparse))
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
                f"Unable to add debug information on input {k!r}.")

        if k not in names_output:
            raise RuntimeError(
                "Unable to find result %r in the ONNX graph. Available="
                "[%s]." % (k, ", ".join(sorted(names_output))))

        index, node = names_output[k]
        new_name = k + suffix

        if id(node) not in nodes_copy:
            new_node = make_node(
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
            inserted_node = make_node(
                node_type, [new_name], [k], domain=domain,
                **atts)
        else:
            pname = k if param_name is None else param_name(k)
            pname += suffix + 'i'
            inserted_node = make_node(
                node_type, [new_name, pname], [k], domain=domain)
            inits.append(from_array(v, name=pname))

        order[id(inserted_node)] = order[id(node)] + 1. / (index + 2)
        nodes[id(inserted_node)] = inserted_node

    new_nodes = [(order[id(n)], n)
                 for n in nodes.values() if id(n) not in nodes_copy]
    new_nodes.extend((order[id(n)], n) for n in nodes_copy.values())
    new_nodes = [n[1] for n in sorted(new_nodes)]

    graph = make_graph(new_nodes, model.graph.name, inputs, outputs,
                       inits, sparse_initializer=inits_sparse)
    onnx_model = make_model(graph, functions=model.functions)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:  # pragma: no cover
        values = {p.key: p.value for p in model.metadata_props}
        set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
    op_set.domain = domain
    op_set.version = domain_opset
    return onnx_model


def onnx_model_to_function(onx, name=None, domain="custom",
                           opset_imports=None, doc_string=None,
                           inputs2par=None):
    """
    Converts an ONNX model into a function. The returned function
    has no attribute.

    :param onx: onnx model
    :param name: function name
    :param domain: function domain
    :param opset_imports: opset to import as a dictionary
        `{domain: version}`
    :param doc_string: doc string
    :param inputs2par: dictionary to move some inputs as attributes
        `{ name: None or default value }`
    :return: function, other functions

    .. warning::
        :epkg:`FunctionProto` does not support default values yet.
        They are ignored.
    """
    if isinstance(onx, ModelProto):
        if opset_imports is None:
            domains = {}
            for op in onx.opset_import:
                domains[op.domain] = op.version
            opset_imports = domains
        if doc_string is None:
            doc_string = onx.doc_string
        fp, lf = onnx_model_to_function(
            onx.graph, name=name, domain=domain,
            opset_imports=opset_imports, doc_string=doc_string,
            inputs2par=inputs2par)
        return fp, lf + list(onx.functions)

    if not isinstance(onx, GraphProto):
        raise TypeError(  # pragma: no cover
            f"Unexpected type {type(onx)!r} for onx.")

    if name is None:
        name = onx.name

    inputs = []
    outputs = [o.name for o in onx.output]
    attributes = []
    nodes = []
    if inputs2par is None:
        inputs.extend(i.name for i in onx.input)
    else:
        for i in onx.input:
            if i.name not in inputs2par:
                inputs.append(i.name)
                continue
            attributes.append(i.name)

    if len(onx.initializer) > 0 or len(onx.sparse_initializer) > 0:
        # Needs to convert every initializer into Constant.
        csts = []
        for init in onx.initializer:
            v = _var_as_dict(init)
            value = from_array(v['value'])
            n = make_node('Constant', [], [init.name], value=value)
            csts.append(n)
        for init in onx.sparse_initializer:
            v = _var_as_dict(init)
            value = from_array(v['sparse_value'])
            n = make_node('Constant', [], [init.name], sparse_value=value)
            csts.append(n)
        nodes.extend(csts)

    nodes.extend(onx.node)

    if isinstance(opset_imports, dict):
        ops = [make_operatorsetid(k, v) for k, v in opset_imports.items()]
        opset_imports = ops
    return make_function(
        domain, name, inputs, outputs, nodes,
        opset_imports=opset_imports, doc_string=doc_string or '',
        attributes=attributes), []


def _onnx_function_to_model_convert_io(ens, type_info, shape_fct):
    typed_io = []
    for name in ens:
        if isinstance(type_info, dict):
            res = type_info[name]
        elif callable(type_info):
            res = type_info(name)
        else:
            raise TypeError(  # pragma: no cover
                "type_info is not a callable or a dictionary, "
                "unable to guess type for name=%r with "
                "type(type_info)=%r." % (name, type(type_info)))
        if isinstance(res, int):
            proto_dtype = res
        else:
            proto_dtype = guess_proto_dtype(res)
        value_info = make_tensor_value_info(
            name, proto_dtype, shape_fct(name, proto_dtype))
        typed_io.append(value_info)
    return typed_io


def onnx_function_to_model(onx, functions=None, type_info=None,
                           as_function=False, shape_fct=None):
    """
    Converts an ONNX FunctionProto into a ModelProto.
    The function does not handle attributes yet.

    :param onx: onnx function
    :param functions: additional functions to append to the model
    :param type_info: dictionary or callable which returns the type of
        inputs or outputs if it cannot be guessed
    :param as_function: if True, the function stays as a function and a single node
        is created to call that function
    :param shape_fct: function to specify the shapes,
        signature: `shape_fct(name, proto_type) -> list`
    :return: function
    """
    if not isinstance(onx, FunctionProto):
        raise TypeError(  # pragma: no cover
            f"onx must be a FunctionProto not {type(onx)!r}.")
    if len(onx.attribute) > 0:
        raise NotImplementedError(
            "The function has attributes, it is not implemented yet.")

    if isinstance(functions, list):
        added_functions = functions.copy()
    elif isinstance(functions, dict):
        added_functions = list(functions.values())
    elif functions is None:
        added_functions = []
    else:
        raise TypeError(  # pragma: no cover
            f"Unexpected type for functions {type(functions)!r}.")

    if shape_fct is None:
        shape_fct = lambda name, dtype: None

    inputs = _onnx_function_to_model_convert_io(
        onx.input, type_info, shape_fct=shape_fct)
    outputs = _onnx_function_to_model_convert_io(
        onx.output, type_info, shape_fct=shape_fct)
    if as_function:
        nodes = [make_node(onx.name,
                           [i.name for i in inputs],
                           [o.name for o in outputs],
                           domain=onx.domain)]
        added_functions.append(onx)
        opsets = [make_operatorsetid(onx.domain, 1)]
    else:
        nodes = list(onx.node)
        opsets = [make_operatorsetid(op.domain, op.version)
                  for op in onx.opset_import]
    graph = make_graph(nodes, onx.name, inputs, outputs,
                       [], doc_string=onx.doc_string)
    model = make_model(graph, functions=added_functions,
                       opset_imports=opsets,
                       doc_string=onx.doc_string,
                       model_version=1,
                       domain=onx.domain)
    return model


def _get_new_name(prefix, name, existing_names):
    opt = f"{prefix}_{name}_0"
    i = 0
    while opt in existing_names:
        i += 1
        opt = "%s_%s_%d" % (prefix, name, i)
    existing_names.add(opt)
    return opt


def onnx_subgraphs_level(obj):
    """
    Returns the depth of the graph.

    :param obj: onnx object
    :return: integer
    """
    if isinstance(obj, ModelProto):
        return onnx_subgraphs_level(obj.graph)
    best = 0
    for node in obj.node:
        for att in node.attribute:
            if (att.type == AttributeProto.GRAPH and
                    hasattr(att, 'g') and att.g is not None):
                m = onnx_subgraphs_level(att.g)
                if m > best:
                    best = m
    return best + 1


class _inline_mapping(dict):
    """
    Overwrites class dictionary to debug more easily.

    :param verbose: verbosity
    :param fLOG: logging function
    :param level: sub graph level
    """

    def __init__(self, verbose, fLOG, level):
        dict.__init__(self)
        self._verbose = verbose
        self._fLOG = fLOG
        self._level = level

    def __setitem__(self, key, value):
        "Adds a value."
        if self._verbose > 3:
            self._fLOG("[_inline_mapping-dict-addkv] %s + %r: %r" %
                       ("  " * self._level, key, value))
        if key in self:
            raise RuntimeError(  # pragma: no cover
                "Key %r was already added (with value %r, new one is %r)."
                "" % (key, self[key], value))
        dict.__setitem__(self, key, value)

    def update(self, d):
        "Updates many values."
        for k, v in d.items():
            self[k] = v

    def copy(self):
        "Returns a copy."
        m = _inline_mapping(self._verbose, self._fLOG, self._level)
        for k, v in self.items():
            m[k] = v
        return m

    def remove(self, o):
        "Removes one element."
        if o not in self:
            raise KeyError(  # pragma: no cover
                f"Cannot remove a key {o!r}.")
        self.pop(o)


def _onnx_inline_function_graph(graph, protos, existing_names, mapping,
                                verbose, fLOG, rename, level):
    if len(graph.node) == 0:
        # Outputs have still to be renamed.
        graph0 = graph
        if verbose > 1:
            fLOG(  # pragma: no cover
                "[onnx_inline_function-graph] %s visit0 graph=%d rename=%r "
                "len(mapping)=%d begin" % (
                    "  " * level, id(graph), rename, len(mapping)))
        if rename:
            modified_nodes = []
            mapping = mapping.copy()
            for i in graph.input:
                mapping[i.name] = i.name
            for i in graph.initializer:
                mapping[i.name] = i.name
            for i in graph.sparse_initializer:
                mapping[i.name] = i.name
            outputs = []
            for o in graph.output:
                no = make_value_info(mapping[o.name], o.type)
                if no.name != o.name:
                    modified_nodes.append(o)
                    outputs.append(no)
                else:
                    outputs.append(o)
            if len(modified_nodes) > 0:
                graph = make_graph(
                    [], graph.name, graph.input, outputs,
                    graph.initializer, doc_string=graph.doc_string,
                    sparse_initializer=list(graph.sparse_initializer))
        else:
            modified_nodes = []

        if verbose > 1:
            fLOG(  # pragma: no cover
                "[onnx_inline_function-graph] %s visit graph=%d end "
                "changed=%r len(modified_nodes)=%d" % (
                    "  " * level, id(graph0), id(graph0) != id(graph),
                    len(modified_nodes)))

        return graph, modified_nodes

    graph0 = graph
    mapping = mapping.copy()
    init = list(graph.initializer)
    init_sparse = list(graph.sparse_initializer)
    inputs = list(graph.input)
    modified_nodes = []
    outputs = list(graph.output)

    if verbose > 1:
        fLOG("[onnx_inline_function-graph] %s >visit graph=%d rename=%r "
             "len(mapping)=%d begin" % (
                 "  " * level, id(graph), rename, len(mapping)))

    output_names = [o.name for o in outputs]
    for i in init:
        mapping[i.name] = i.name
    for i in init_sparse:
        mapping[i.name] = i.name
    for i in inputs:
        mapping[i.name] = i.name

    # first step, replace names
    nodes = []
    for node in graph.node:
        mod = 0
        inp = []
        for i in node.input:
            if i in mapping:
                inp.append(mapping[i])
                if mapping[i] != i:
                    mod += 1
            else:
                raise RuntimeError(  # pragma: no cover
                    "Cannot find input %r in %s for node (level=%d)\n%r." % (
                        i, pprint.pformat(mapping), level, node))
        out = []
        for o in node.output:
            new_o = o
            if rename:
                if o not in output_names:
                    new_o = _get_new_name('_inl', o, existing_names)
                if o in mapping:
                    # See below.
                    mapping.remove(o)
            elif o in mapping:
                # That means the main contains a result node but is overwritten by
                # the subgraph. The local variable cannot be reached anymore,
                # we remove it.
                mapping.remove(o)
                if o in node.input:
                    new_o = _get_new_name('_inl', o, existing_names)
                if verbose > 3:
                    fLOG(
                        "[onnx_inline_function-renam] %s node %r(%r): %r -> %r "
                        "overwrite result (%r -> %r)." % (
                            "  " * level, node.op_type, node.name, node.input,
                            node.output, o, new_o))
            out.append(new_o)
            mapping[o] = new_o
            if o != new_o:
                mapping[new_o] = new_o
                mod += 1

        if verbose > 3:
            fLOG("[onnx_inline_function-renam] %s rep node %r(%r): %r -> %r" % (
                "  " * level, node.op_type, node.name, node.input, node.output))
        new_node = make_node(node.op_type, inp, out, domain=node.domain,
                             name=_get_new_name('_inln', node.name, existing_names))
        for att in node.attribute:
            if (att.type == AttributeProto.GRAPH and
                    hasattr(att, 'g') and att.g is not None):
                g, m = _onnx_inline_function_graph(
                    att.g, protos, existing_names=existing_names,
                    verbose=verbose, fLOG=fLOG, mapping=mapping,
                    rename=rename, level=level + 1)
                if len(m) > 0:
                    att = make_attribute(att.name, g)
                    mod += len(m)
                else:
                    att = make_attribute(att.name, att.g)
            new_node.attribute.append(att)
        if mod > 0:
            if verbose > 2:
                fLOG("[onnx_inline_function-renam] %s add node %r(%r): %r -> %r" % (
                    "  " * level,
                    new_node.op_type, new_node.name,
                    new_node.input, new_node.output))
            nodes.append(new_node)
            modified_nodes.append(node)
        else:
            nodes.append(node)

    if len(modified_nodes) > 0:
        if verbose > 1:
            fLOG("[onnx_inline_function-graph] %s -1 graph=%d "
                 "len(modified_nodes)=%d" % (
                     "  " * level, id(graph), len(modified_nodes)))

        graph = make_graph(
            nodes, graph.name, inputs, outputs,
            init, doc_string=graph.doc_string,
            sparse_initializer=list(graph.sparse_initializer))
    elif not rename:
        # no modification, let's check the node hiding a functions
        new_nodes = []
        for node in nodes:
            nnodes, m = _onnx_inline_function_node(
                node, protos, existing_names, verbose, fLOG,
                level=level)
            if len(m) > 0:
                if verbose > 0:
                    fLOG("[onnx_inline_function-subgr] %s replaced node %r (%r) "
                         "with %d nodes (id=%r) -- %r -> %r" % (
                             "  " * level,
                             node.name, node.op_type, len(nnodes), id(node),
                             node.input, node.output))
                new_nodes.extend(nnodes)
                modified_nodes.extend(m)
            else:
                new_nodes.append(node)
        if len(modified_nodes) > 0:
            if verbose > 1:
                fLOG("[onnx_inline_function-graph] %s -2 graph=%d "
                     "len(modified_nodes)=%d" % (
                         "  " * level, id(graph), len(modified_nodes)))

            nodes = new_nodes
            graph = make_graph(
                nodes, graph.name, inputs, outputs,
                init, doc_string=graph.doc_string,
                sparse_initializer=list(graph.sparse_initializer))

    if verbose > 1:
        fLOG("[onnx_inline_function-graph] %s <visit graph=%d end "
             "changed=%r len(modified_nodes)=%d" % (
                 "  " * level, id(graph0), id(graph0) != id(graph),
                 len(modified_nodes)))

    return graph, modified_nodes


def _onnx_inline_function_node(node, protos, existing_names, verbose,
                               fLOG, level):
    # The function does not rename input or output
    # of the node, it just replaces the node but a function
    # if the function exists.
    modified_nodes = []
    key = node.domain, node.op_type
    if key in protos:
        proto = protos[key]
        if not isinstance(proto, FunctionProto):
            raise TypeError(  # pragma: no cover
                "Prototype for key=%r must be a Function Proto, not %r." % (
                    key, type(proto)))
        modified_nodes.append(node)
        new_nodes = []
        mapping = _inline_mapping(verbose, fLOG, level)
        prefix = "_inl"

        for fr, to in zip(node.input, proto.input):
            n = make_node('Identity', [fr],
                          [_get_new_name(prefix, to, existing_names)])
            if verbose > 2:
                fLOG("[onnx_inline_function-ninpu] %s add node %r(%r): %r -> %r" % (
                    "  " * level, n.op_type, n.name, n.input, n.output))
            mapping[to] = n.output[0]
            if to != n.output[0]:
                mapping[n.output[0]] = n.output[0]
            new_nodes.append(n)

        for nn in proto.node:
            new_input = [mapping[i] for i in nn.input]
            new_output = [_get_new_name(prefix, o, existing_names)
                          for o in nn.output]
            mapping.update(
                {o: oo for o, oo in zip(nn.output, new_output)})
            mapping.update({oo: oo for oo in new_output})
            new_node = make_node(
                nn.op_type, new_input, new_output,
                domain=nn.domain, name=_get_new_name(
                    prefix, nn.name, existing_names))
            if verbose > 3:
                fLOG("[onnx_inline_function-nnode] %s rep node %r(%r): %r -> %r" % (
                    "  " * level, nn.op_type, nn.name, nn.input, nn.output))
            if verbose > 2:
                fLOG("[onnx_inline_function-nnode] %s add node %r(%r): %r -> %r" % (
                    "  " * level,
                    new_node.op_type, new_node.name,
                    new_node.input, new_node.output))
            for att in nn.attribute:
                if (att.type == AttributeProto.GRAPH and
                        hasattr(att, 'g') and att.g is not None):
                    if verbose > 1:
                        fLOG("[onnx_inline_function-funct] %s fct=%r graph=%d node=%d" % (
                            "  " * level, key, id(att.g), id(new_node)))

                    g, m = _onnx_inline_function_graph(
                        att.g, protos, existing_names=existing_names,
                        verbose=verbose, fLOG=fLOG, mapping=mapping,
                        rename=True, level=level + 1)
                    if len(m) > 0:
                        att = make_attribute(att.name, g)
                    else:
                        att = make_attribute(att.name, att.g)
                new_node.attribute.append(att)
            new_nodes.append(new_node)

        for fr, to in zip(proto.output, node.output):
            n = make_node('Identity', [mapping[fr]], [to])
            if verbose > 2:
                fLOG("[onnx_inline_function-noutt] %s add node %r(%r): %r -> %r" % (
                    "  " * level, n.op_type, n.name, n.input, n.output))
            new_nodes.append(n)
    else:
        new_nodes = [node]
        modified_nodes = []
    return new_nodes, modified_nodes


def onnx_inline_function(obj, protos=None, existing_names=None, verbose=0, fLOG=None):
    """
    Inlines functions in an ONNX graph.

    :param obj: onnx graph, :epkg:`FunctionProto`, :epkg:`GraphProto`,
        :epkg:`ModelProto`
    :param protos: if None, the function assumes *obj* is of type
        :epkg:`ModelProto` and the goal is to inline every function.
        If *protos* a list of strings, the function only inlines the
        functions in that list. If *protos* is a dictionary
        `{ (domain, type): FunctionProto }`, the function replaces every
        node `(domain, type)` by the code given in this dictionary
    :param existing_names: no new name will be taken in that set
    :param verbose: verbosity
    :param fLOG: logging function
    :return: modified object, list of modified nodes

    .. versionadded:: 0.9
    """
    if verbose > 0 and fLOG is None:
        fLOG = print
    if isinstance(obj, ModelProto):
        if verbose > 0:
            fLOG("[onnx_inline_function] type=%r graph=%d" % (
                type(obj), id(obj)))
        if protos is None:
            fct = [f.name for f in obj.functions]
            ex_names = set(enumerate_onnx_names(obj))
            if existing_names is not None:
                ex_names |= existing_names
            return onnx_inline_function(obj, fct, existing_names=ex_names,
                                        verbose=verbose, fLOG=fLOG)
        if isinstance(protos, list):
            ex_names = set(enumerate_onnx_names(obj))
            if existing_names is not None:
                ex_names |= existing_names
            protos = {(f.domain, f.name): f for f in obj.functions}
            return onnx_inline_function(obj, protos, existing_names=ex_names,
                                        verbose=verbose, fLOG=fLOG)
    if isinstance(protos, list):
        protos = {(f.domain, f.name): f for f in protos}
    if not isinstance(protos, dict):
        raise TypeError(  # pragma: no cover
            "obj is of type %r and protos must be a dictionary not %r." % (
                type(obj), type(protos)))

    if isinstance(obj, ModelProto):
        new_graph, m = onnx_inline_function(
            obj.graph, protos, verbose=verbose, fLOG=fLOG)
        if len(new_graph.initializer) != len(obj.graph.initializer):
            raise RuntimeError(  # pragma: no cover
                "Mismatched number of initializers %d != %d." % (
                    len(new_graph.initializer), len(obj.graph.initializer)))
        if len(new_graph.sparse_initializer) != len(obj.graph.sparse_initializer):
            raise RuntimeError(  # pragma: no cover
                "Mismatched number of initializers %d != %d." % (
                    len(new_graph.sparse_initializer),
                    len(obj.graph.sparse_initializer)))
        new_functions = []
        distri = Counter(
            (n.domain, n.op_type)
            for n in enumerate_onnx_nodes(new_graph))
        opsets = {op.domain: op.version for op in obj.opset_import}
        for f in obj.functions:
            key = f.domain, f.name
            if key not in protos:
                new_functions.append(f)
            elif key in distri:
                raise RuntimeError(  # pragma: no cover
                    "Function %r still appears in the graph, "
                    "distibution=%s." % (key, pprint.pformat(distri)))
            if f.domain not in opsets:
                opsets[f.domain] = 1
        return (
            make_model(
                new_graph,
                functions=new_functions,
                opset_imports=[
                    make_operatorsetid(k, v)
                    for k, v in opsets.items()],
                producer_name=obj.producer_name,
                producer_version=obj.producer_version,
                ir_version=obj.ir_version,
                doc_string=obj.doc_string,
                domain=obj.domain,
                model_version=obj.model_version),
            m)

    # FunctionProto, GraphProto
    if existing_names is None:
        existing_names = set(enumerate_onnx_names(obj))

    if verbose > 0:
        fLOG("[onnx_inline_function] type=%r graph=%d begin" % (
            type(obj), id(obj)))
        distri = Counter((n.domain, n.op_type)
                         for n in enumerate_onnx_nodes(obj))

    new_nodes = list(obj.node)
    modified_nodes = []
    n_iter = 0
    max_iter = onnx_subgraphs_level(obj) + 1
    modified = 1
    while modified > 0 and n_iter < max_iter:
        if verbose > 0:
            fLOG(f"[onnx_inline_function] start iteration {n_iter!r}")

        # local context
        mapping = _inline_mapping(verbose, fLOG, level=0)
        if isinstance(obj, GraphProto):
            mapping.update({i.name: i.name for i in obj.initializer})
            mapping.update({i.name: i.name for i in obj.sparse_initializer})
            for i in obj.input:
                if i.name not in mapping:
                    mapping[i.name] = i.name
        elif isinstance(obj, FunctionProto):
            mapping.update({i: i for i in obj.input})
        else:
            raise TypeError(  # pragma: no cover
                f"Unexpected type for obj: {type(obj)!r}.")

        # loop on nodes
        old_nodes = new_nodes
        modified = 0
        new_nodes = []
        for node in old_nodes:
            nnodes, m = _onnx_inline_function_node(
                node, protos, existing_names, verbose, fLOG, level=0)
            mapping.update({o: o for o in node.output})

            if len(m) > 0:
                if verbose > 0:
                    fLOG("[onnx_inline_function] replaced node %r (%r) "
                         "with %d nodes (id=%r) -- %r -> %r (iter=%r)" % (
                             node.name, node.op_type, len(nnodes), id(node),
                             node.input, node.output, n_iter))
                modified += len(m)
                new_nodes.extend(nnodes)
                modified_nodes.extend(m)
            else:
                has_graph = False
                new_attributes = []
                for att in node.attribute:
                    if (att.type == AttributeProto.GRAPH and
                            hasattr(att, 'g') and att.g is not None):
                        g, m = _onnx_inline_function_graph(
                            att.g, protos, verbose=verbose, fLOG=fLOG,
                            existing_names=existing_names, mapping=mapping,
                            rename=False, level=1)
                        if len(m) > 0:
                            modified_nodes.extend(m)
                            modified_nodes.append(node)
                            modified += 1 + len(m)
                            has_graph = True
                            att = make_attribute(att.name, g)
                    new_attributes.append(att)
                if has_graph:
                    new_node = make_node(
                        node.op_type, node.input, node.output,
                        domain=node.domain, name=node.name)
                    new_node.attribute.extend(new_attributes)
                    new_nodes.append(new_node)
                else:
                    # we still need to check that this subgraph does
                    # not include a function
                    new_nodes.append(node)

        n_iter += 1
        if verbose > 0:
            total_node = len(list(enumerate_onnx_nodes(new_nodes)))
            fLOG("[onnx_inline_function] n_iter=%r/%r nodes=%r modified=%r "
                 "n_nodes=%d total=%d" % (
                     n_iter, max_iter, len(obj.node), modified,
                     len(new_nodes), total_node))

    if verbose > 0:
        fLOG("[onnx_inline_function] type=%r graph=%d end with %d "
             "modified nodes" % (
                 type(obj), id(obj), len(modified_nodes)))
        distri2 = Counter((n.domain, n.op_type)
                          for n in enumerate_onnx_nodes(new_nodes))
        if distri != distri2:
            fLOG("[onnx_inline_function] BEFORE")
            for k, v in sorted(distri.items()):
                fLOG("[onnx_inline_function] %d -- %s" % (v, k))
            fLOG("[onnx_inline_function] AFTER")
            for k, v in sorted(distri2.items()):
                fLOG("[onnx_inline_function] %d -- %s" % (v, k))

    if isinstance(obj, FunctionProto):
        return (
            make_function(
                domain=obj.domain, fname=obj.name,
                inputs=obj.input, outputs=obj.output, nodes=new_nodes,
                opset_imports=[
                    make_operatorsetid(op.domain, op.version)
                    for op in obj.opset_import],
                doc_string=obj.doc_string,
                attributes=obj.attribute),
            modified_nodes)
    if isinstance(obj, GraphProto):
        return (
            make_graph(new_nodes, obj.name, list(obj.input), list(obj.output),
                       list(obj.initializer), doc_string=obj.doc_string,
                       sparse_initializer=list(obj.sparse_initializer)),
            modified_nodes)
    raise TypeError(  # pragma: no cover
        f"Unexpected type for obj {type(obj)!r}.")
