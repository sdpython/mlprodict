# pylint: disable=R0912
"""
@file
@brief Text representations of graphs.
"""
from collections import OrderedDict
import numpy
from onnx import TensorProto, AttributeProto
from onnx.numpy_helper import to_array
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from ..tools.graphs import onnx2bigraph
from ..onnx_tools.onnx2py_helper import _var_as_dict


def onnx_text_plot(model_onnx, recursive=False, graph_type='basic',
                   grid=5, distance=5):
    """
    Uses @see fn onnx2bigraph to convert the ONNX graph
    into text.

    :param model_onnx: onnx representation
    :param recursive: @see fn onnx2bigraph
    :param graph_type: @see fn onnx2bigraph
    :param grid: @see me display_structure
    :param distance: @see fn display_structure
    :return: text

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import numpy
        from mlprodict.onnx_conv import to_onnx
        from mlprodict import __max_supported_opset__ as opv
        from mlprodict.plotting.plotting import onnx_text_plot
        from mlprodict.npy.xop import loadop

        OnnxAdd, OnnxSub = loadop('Add', 'Sub')

        idi = numpy.identity(2).astype(numpy.float32)
        A = OnnxAdd('X', idi, op_version=opv)
        B = OnnxSub(A, 'W', output_names=['Y'], op_version=opv)
        onx = B.to_onnx({'X': idi, 'W': idi})
        print(onnx_text_plot(onx))
    """
    bigraph = onnx2bigraph(model_onnx)
    graph = bigraph.display_structure()
    return graph.to_text()


def onnx_text_plot_tree(node):
    """
    Gives a textual representation of a tree ensemble.

    :param node: `TreeEnsemble*`
    :return: text

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import numpy
        from sklearn.datasets import load_iris
        from sklearn.tree import DecisionTreeRegressor
        from mlprodict.onnx_conv import to_onnx
        from mlprodict.plotting.plotting import onnx_text_plot_tree

        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        clr = DecisionTreeRegressor(max_depth=3)
        clr.fit(X, y)
        onx = to_onnx(clr, X)
        res = onnx_text_plot_tree(onx.graph.node[0])
        print(res)
    """
    def rule(r):
        if r == b'BRANCH_LEQ':
            return '<='
        if r == b'BRANCH_LT':  # pragma: no cover
            return '<'
        if r == b'BRANCH_GEQ':  # pragma: no cover
            return '>='
        if r == b'BRANCH_GT':  # pragma: no cover
            return '>'
        if r == b'BRANCH_EQ':  # pragma: no cover
            return '=='
        if r == b'BRANCH_NEQ':  # pragma: no cover
            return '!='
        raise ValueError(  # pragma: no cover
            "Unexpected rule %r." % rule)

    class Node:
        "Node representation."

        def __init__(self, i, atts):
            self.nodes_hitrates = None
            self.nodes_missing_value_tracks_true = None
            for k, v in atts.items():
                if k.startswith('nodes'):
                    setattr(self, k, v[i])
            self.depth = 0
            self.true_false = ''

        def process_node(self):
            "node to string"
            if self.nodes_modes == b'LEAF':  # pylint: disable=E1101
                text = "%s y=%r f=%r i=%r" % (
                    self.true_false,
                    self.target_weights, self.target_ids,  # pylint: disable=E1101
                    self.target_nodeids)  # pylint: disable=E1101
            else:
                text = "%s X%d %s %r" % (
                    self.true_false,
                    self.nodes_featureids,  # pylint: disable=E1101
                    rule(self.nodes_modes),  # pylint: disable=E1101
                    self.nodes_values)  # pylint: disable=E1101
                if self.nodes_hitrates and self.nodes_hitrates != 1:
                    text += " hi=%r" % self.nodes_hitrates
                if self.nodes_missing_value_tracks_true:
                    text += " miss=%r" % (
                        self.nodes_missing_value_tracks_true)
            return "%s%s" % ("   " * self.depth, text)

    def process_tree(atts, treeid):
        "tree to string"
        rows = ['treeid=%r' % treeid]
        if 'base_values' in atts:
            rows.append('base_value=%r' % atts['base_values'][treeid])

        short = {}
        for prefix in ['nodes', 'target', 'class']:
            if ('%s_treeids' % prefix) not in atts:
                continue
            idx = [i for i in range(len(atts['%s_treeids' % prefix]))
                   if atts['%s_treeids' % prefix][i] == treeid]
            for k, v in atts.items():
                if k.startswith(prefix):
                    short[k] = [v[i] for i in idx]

        nodes = OrderedDict()
        for i in range(len(short['nodes_treeids'])):
            nodes[i] = Node(i, short)
        for i in range(len(short['target_treeids'])):
            idn = short['target_nodeids'][i]
            node = nodes[idn]
            node.target_nodeids = idn
            node.target_ids = short['target_ids'][i]
            node.target_weights = short['target_weights'][i]

        def iterate(nodes, node, depth=0, true_false=''):
            node.depth = depth
            node.true_false = true_false
            yield node
            if node.nodes_falsenodeids > 0:
                for n in iterate(nodes, nodes[node.nodes_falsenodeids],
                                 depth=depth + 1, true_false='F'):
                    yield n
                for n in iterate(nodes, nodes[node.nodes_truenodeids],
                                 depth=depth + 1, true_false='T'):
                    yield n

        for node in iterate(nodes, nodes[0]):
            rows.append(node.process_node())
        return rows

    if node.op_type != "TreeEnsembleRegressor":
        raise NotImplementedError(  # pragma: no cover
            "Type %r cannot be displayed." % node.op_type)
    d = {k: v['value'] for k, v in _var_as_dict(node)['atts'].items()}
    atts = {}
    for k, v in d.items():
        atts[k] = v if isinstance(v, int) else list(v)
    trees = list(sorted(set(atts['nodes_treeids'])))
    rows = ['n_targets=%r' % atts['n_targets'],
            'n_trees=%r' % len(trees)]
    for tree in trees:
        r = process_tree(atts, tree)
        rows.append('----')
        rows.extend(r)

    return "\n".join(rows)


def reorder_nodes_for_display(nodes, verbose=False):
    """
    Reorders the node with breadth first seach (BFS).

    :param nodes: list of ONNX nodes
    :param verbose: dislay intermediate informations
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
        node_name = node.name + "#" + "|".join(node.output)
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
        inputs = dnodes[node_name].input
        if any(map(lambda i: i not in known, inputs)):
            return []

        res = [node_name]
        while res[-1] in successors:
            next_names = successors[res[-1]]
            if res[-1] not in dnodes:
                next_names = set(v for v in next_names if v not in known)
                if len(next_names) == 1:
                    next_name = next_names.pop()
                    inputs = dnodes[next_name].input
                    if any(map(lambda i: i not in known, inputs)):
                        break
                    res.extend(next_name)
                else:
                    break
            else:
                next_names = set(v for v in next_names if v not in done)
                if len(next_names) == 1:
                    next_name = next_names.pop()
                    res.append(next_name)
                else:
                    break

        return [r for r in res if r in dnodes and r not in done]

    while len(done) < len(nodes):
        # possible
        possibles = OrderedDict()
        for k, v in dnodes.items():
            if k in done:
                continue
            if predecessors[k] <= known:
                possibles[k] = v

        sequences = OrderedDict()
        for k, v in possibles.items():
            if k in done:
                continue
            sequences[k] = _find_sequence(k, known, done)
            if verbose:
                print("[reorder_nodes_for_display] sequence(%s)=%s" % (
                    k, ",".join(sequences[k])))

        if len(sequences) == 0:
            raise RuntimeError(  # pragma: no cover
                "Unexpected empty sequences (len(possibles)=%d, "
                "len(done)=%d, len(nodes)=%d). This is usually due to "
                "a name used both as result name and node node."
                "" % (len(possibles), len(done), len(nodes)))

        # find the best sequence
        best = None
        for k, v in sequences.items():
            if best is None or len(v) > len(sequences[best]):
                # if the sequence of successors is longer
                best = k
            elif len(v) == len(sequences[best]):
                if len(new_nodes) > 0:
                    # then choose the next successor sharing input with
                    # previous output
                    so = set(new_nodes[-1].output)
                    first1 = dnodes[sequences[best][0]]
                    first2 = dnodes[v[0]]
                    if len(set(first1.input) & so) < len(set(first2.input) & so):
                        best = k
                else:
                    first1 = dnodes[sequences[best][0]]
                    first2 = dnodes[v[0]]
                    if first1.op_type > first2.op_type:
                        best = k
                    elif (first1.op_type == first2.op_type and
                            first1.name > first2.name):
                        best = k

        if best is None:
            raise RuntimeError(  # pragma: no cover
                "Wrong implementation (len(sequence)=%d)." % len(sequences))
        if verbose:
            print("[reorder_nodes_for_display] BEST: sequence(%s)=%s" % (
                best, ",".join(sequences[best])))

        # process the sequence
        for k in sequences[best]:
            v = dnodes[k]
            new_nodes.append(v)
            done.add(k)
            known |= set(v.output)

    if len(new_nodes) != len(nodes):
        raise RuntimeError(  # pragma: no cover
            "The returned new nodes are different. "
            "len(nodes=%d != %d=len(new_nodes). done=\n%r"
            "\n%s\n----------\n%s" % (
                len(nodes), len(new_nodes), done,
                "\n".join("%d - %s - %s - %s" % (
                    (n.name + "".join(n.output)) in done,
                    n.op_type, n.name, n.name + "".join(n.output))
                    for n in nodes),
                "\n".join("%d - %s - %s - %s" % (
                    (n.name + "".join(n.output)) in done,
                    n.op_type, n.name, n.name + "".join(n.output))
                    for n in new_nodes)))
    return new_nodes


def _get_type(obj0):
    obj = obj0
    if hasattr(obj, 'data_type'):
        if (obj.data_type == TensorProto.FLOAT and  # pylint: disable=E1101
                hasattr(obj, 'float_data')):
            return TENSOR_TYPE_TO_NP_TYPE[TensorProto.FLOAT]  # pylint: disable=E1101
        if (obj.data_type == TensorProto.DOUBLE and  # pylint: disable=E1101
                hasattr(obj, 'double_data')):
            return TENSOR_TYPE_TO_NP_TYPE[TensorProto.DOUBLE]  # pylint: disable=E1101
        if (obj.data_type == TensorProto.INT64 and  # pylint: disable=E1101
                hasattr(obj, 'int64_data')):
            return TENSOR_TYPE_TO_NP_TYPE[TensorProto.INT64]  # pylint: disable=E1101
        if (obj.data_type == TensorProto.INT32 and  # pylint: disable=E1101
                hasattr(obj, 'int32_data')):
            return TENSOR_TYPE_TO_NP_TYPE[TensorProto.INT32]  # pylint: disable=E1101
        raise RuntimeError(  # pragma: no cover
            "Unable to guess type from %r." % obj0)
    if hasattr(obj, 'type'):
        obj = obj.type
    if hasattr(obj, 'tensor_type'):
        obj = obj.tensor_type
    if hasattr(obj, 'elem_type'):
        return TENSOR_TYPE_TO_NP_TYPE.get(obj.elem_type, '?')
    raise RuntimeError(  # pragma: no cover
        "Unable to guess type from %r." % obj0)


def _get_shape(obj):
    obj0 = obj
    if hasattr(obj, 'data_type'):
        if (obj.data_type == TensorProto.FLOAT and  # pylint: disable=E1101
                hasattr(obj, 'float_data')):
            return (len(obj.float_data), )
        if (obj.data_type == TensorProto.DOUBLE and  # pylint: disable=E1101
                hasattr(obj, 'double_data')):
            return (len(obj.double_data), )
        if (obj.data_type == TensorProto.INT64 and  # pylint: disable=E1101
                hasattr(obj, 'int64_data')):
            return (len(obj.int64_data), )
        if (obj.data_type == TensorProto.INT32 and  # pylint: disable=E1101
                hasattr(obj, 'int32_data')):
            return (len(obj.int32_data), )
        raise RuntimeError(  # pragma: no cover
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
    raise RuntimeError(  # pragma: no cover
        "Unable to guess type from %r." % obj0)


def onnx_simple_text_plot(model, verbose=False, att_display=None,
                          add_links=False, recursive=False, functions=True):
    """
    Displays an ONNX graph into text.

    :param model: ONNX graph
    :param verbose: display debugging information
    :param att_display: list of attributes to display, if None,
        a default list if used
    :param add_links: displays links of the right side
    :param recursive: display subgraphs as well
    :param functions: display functions as well
    :return: str

    An ONNX graph is printed the following way:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import numpy
        from sklearn.cluster import KMeans
        from mlprodict.plotting.plotting import onnx_simple_text_plot
        from mlprodict.onnx_conv import to_onnx

        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = KMeans(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_simple_text_plot(onx, verbose=False)
        print(text)

    The same graphs with links.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import numpy
        from sklearn.cluster import KMeans
        from mlprodict.plotting.plotting import onnx_simple_text_plot
        from mlprodict.onnx_conv import to_onnx

        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = KMeans(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_simple_text_plot(onx, verbose=False, add_links=True)
        print(text)

    Visually, it looks like the following:

    .. gdot::
        :script: DOT-SECTION

        import numpy
        from sklearn.cluster import KMeans
        from mlprodict.onnxrt import OnnxInference
        from mlprodict.onnx_conv import to_onnx

        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = KMeans(3)
        model.fit(x, y)
        model_onnx = to_onnx(model, x.astype(numpy.float32),
                             target_opset=15)
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    if att_display is None:
        att_display = [
            'activations',
            'align_corners',
            'allowzero',
            'alpha',
            'auto_pad',
            'axis',
            'axes',
            'batch_axis',
            'batch_dims',
            'beta',
            'bias',
            'blocksize',
            'case_change_action',
            'ceil_mode',
            'center_point_box',
            'clip',
            'coordinate_transformation_mode',
            'count_include_pad',
            'cubic_coeff_a',
            'decay_factor',
            'detect_negative',
            'detect_positive',
            'dilation',
            'dilations',
            'direction',
            'dtype',
            'end',
            'epsilon',
            'equation',
            'exclusive',
            'exclude_outside',
            'extrapolation_value',
            'fmod',
            'gamma',
            'group',
            'hidden_size',
            'high',
            'ignore_index',
            'input_forget',
            'is_case_sensitive',
            'k',
            'keepdims',
            'kernel_shape',
            'lambd',
            'largest',
            'layout',
            'linear_before_reset',
            'locale',
            'low',
            'max_gram_length',
            'max_skip_count',
            'mean',
            'min_gram_length',
            'mode',
            'momentum',
            'nearest_mode',
            'ngram_counts',
            'ngram_indexes',
            'noop_with_empty_axes',
            'norm_coefficient',
            'norm_coefficient_post',
            'num_scan_inputs',
            'output_height',
            'output_padding',
            'output_shape',
            'output_width',
            'p',
            'padding_mode',
            'pads',
            'perm',
            'pooled_shape',
            'reduction',
            'reverse',
            'sample_size',
            'sampling_ratio',
            'scale',
            'scan_input_axes',
            'scan_input_directions',
            'scan_output_axes',
            'scan_output_directions',
            'seed',
            'select_last_index',
            'size',
            'sorted',
            'spatial_scale',
            'start',
            'storage_order',
            'strides',
            'time_axis',
            'to',
            'training_mode',
            'transA',
            'transB',
            'type',
            'upper',
            'xs',
            'y',
            'zs',
        ]

    def str_node(indent, node):
        atts = []
        if hasattr(node, 'attribute'):
            for att in node.attribute:
                if att.name in att_display:
                    if att.type == AttributeProto.INT:  # pylint: disable=E1101
                        atts.append("%s=%d" % (att.name, att.i))
                    elif att.type == AttributeProto.FLOAT:  # pylint: disable=E1101
                        atts.append("%s=%1.2f" % (att.name, att.f))
                    elif att.type == AttributeProto.INTS:  # pylint: disable=E1101
                        atts.append("%s=%s" % (att.name, str(
                            list(att.ints)).replace(" ", "")))
        inputs = list(node.input)
        if len(atts) > 0:
            inputs.extend(atts)
        return "%s%s(%s) -> %s" % (
            "  " * indent, node.op_type,
            ", ".join(inputs), ", ".join(node.output))

    rows = []
    if hasattr(model, 'opset_import'):
        for opset in model.opset_import:
            rows.append("opset: domain=%r version=%r" % (
                opset.domain, opset.version))
    if hasattr(model, 'graph'):
        main_model = model
        model = model.graph
    else:
        main_model = None

    # inputs
    line_name_new = {}
    line_name_in = {}
    for inp in model.input:
        if isinstance(inp, str):
            rows.append("input: %r" % inp)
        else:
            line_name_new[inp.name] = len(rows)
            rows.append("input: name=%r type=%r shape=%r" % (
                inp.name, _get_type(inp), _get_shape(inp)))
    # initializer
    if hasattr(model, 'initializer'):
        for init in model.initializer:
            if numpy.prod(_get_shape(init)) < 5:
                content = " -- %r" % to_array(init).ravel()
            else:
                content = ""
            line_name_new[init.name] = len(rows)
            rows.append("init: name=%r type=%r shape=%r%s" % (
                init.name, _get_type(init), _get_shape(init), content))

    # successors, predecessors
    successors = {}
    predecessors = {}
    subgraphs = []
    for node in model.node:
        node_name = node.name + "#" + "|".join(node.output)
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
        if recursive and node.op_type in {'If', 'Scan', 'Loop'}:
            for att in node.attribute:
                if att.name not in {'body', 'else_branch', 'then_branch'}:
                    continue
                subgraphs.append((node, att.name, att.g))

    # walk through nodes
    init_names = set()
    indents = {}
    for inp in model.input:
        if isinstance(inp, str):
            indents[inp] = 0
            init_names.add(inp)
        else:
            indents[inp.name] = 0
            init_names.add(inp.name)
    if hasattr(model, 'initializer'):
        for init in model.initializer:
            indents[init.name] = 0
            init_names.add(init.name)

    nodes = reorder_nodes_for_display(model.node, verbose=verbose)

    previous_indent = None
    previous_out = None
    previous_in = None
    for node in nodes:
        add_break = False
        name = node.name + "#" + "|".join(node.output)
        if name in indents:
            indent = indents[name]
            if previous_indent is not None and indent < previous_indent:
                if verbose:
                    print("[onnx_simple_text_plot] break1 %s" % node.op_type)
                add_break = True
        elif previous_in is not None and set(node.input) == previous_in:
            indent = previous_indent
        else:
            inds = [indents.get(i, 0)
                    for i in node.input if i not in init_names]
            if len(inds) == 0:
                indent = 0
            else:
                mi = min(inds)
                indent = mi
                if previous_indent is not None and indent < previous_indent:
                    if verbose:
                        print(  # pragma: no cover
                            "[onnx_simple_text_plot] break2 %s" %
                                node.op_type)
                    add_break = True
            if not add_break and previous_out is not None:
                if len(set(node.input) & previous_out) == 0:
                    if verbose:
                        print("[onnx_simple_text_plot] break3 %s" %
                              node.op_type)
                    add_break = True
                    indent = 0

        if add_break and verbose:
            print("[onnx_simple_text_plot] add break")
        for n in node.input:
            if n in line_name_in:
                line_name_in[n].append(len(rows))
            else:
                line_name_in[n] = [len(rows)]
        for n in node.output:
            line_name_new[n] = len(rows)
        rows.append(str_node(indent, node))
        indents[name] = indent

        for i, o in enumerate(node.output):
            indents[o] = indent + 1

        previous_indent = indents[name]
        previous_out = set(node.output)
        previous_in = set(node.input)

    # outputs
    for out in model.output:
        if isinstance(out, str):
            if out in line_name_in:
                line_name_in[out].append(len(rows))
            else:
                line_name_in[out] = [len(rows)]
            rows.append("output: name=%r type=%s shape=%s" % (
                out, '?', '?'))
        else:
            if out.name in line_name_in:
                line_name_in[out.name].append(len(rows))
            else:
                line_name_in[out.name] = [len(rows)]
            rows.append("output: name=%r type=%r shape=%r" % (
                out.name, _get_type(out), _get_shape(out)))

    if add_links:

        def _mark_link(rows, lengths, r1, r2, d):
            maxl = max(lengths[r1], lengths[r2]) + d * 2
            maxl = max(maxl, max(len(rows[r]) for r in range(r1, r2 + 1))) + 2

            if rows[r1][-1] == '|':
                p1, p2 = rows[r1][:lengths[r1] + 2], rows[r1][lengths[r1] + 2:]
                rows[r1] = p1 + p2.replace(' ', '-')
            rows[r1] += ("-" * (maxl - len(rows[r1]) - 1)) + "+"

            if rows[r2][-1] == " ":
                rows[r2] += "<"
            elif rows[r2][-1] == '|':
                if "<" not in rows[r2]:
                    p = lengths[r2]
                    rows[r2] = rows[r2][:p] + '<' + rows[r2][p + 1:]
                p1, p2 = rows[r2][:lengths[r2] + 2], rows[r2][lengths[r2] + 2:]
                rows[r2] = p1 + p2.replace(' ', '-')
            rows[r2] += ("-" * (maxl - len(rows[r2]) - 1)) + "+"

            for r in range(r1 + 1, r2):
                if len(rows[r]) < maxl:
                    rows[r] += " " * (maxl - len(rows[r]) - 1)
                rows[r] += "|"

        diffs = []
        for n, r1 in line_name_new.items():
            if n not in line_name_in:
                continue
            r2s = line_name_in[n]
            for r2 in r2s:
                if r1 >= r2:
                    continue
                diffs.append((r2 - r1, (n, r1, r2)))
        diffs.sort()
        for i in range(len(rows)):  # pylint: disable=C0200
            rows[i] += "  "
        lengths = [len(r) for r in rows]

        for d, (n, r1, r2) in diffs:
            if d == 1 and len(line_name_in[n]) == 1:
                # no line for link to the next node
                continue
            _mark_link(rows, lengths, r1, r2, d)

    # subgraphs
    for node, name, g in subgraphs:
        rows.append('----- subgraph ---- %s - %s - att.%s=' % (
            node.op_type, node.name, name))
        res = onnx_simple_text_plot(
            g, verbose=verbose, att_display=att_display,
            add_links=add_links, recursive=recursive)
        rows.append(res)

    # functions
    if functions and main_model is not None:
        for fct in main_model.functions:
            rows.append('----- function name=%s domain=%s' % (
                fct.name, fct.domain))
            res = onnx_simple_text_plot(
                fct, verbose=verbose, att_display=att_display,
                add_links=add_links, recursive=recursive,
                functions=False)
            rows.append(res)

    return "\n".join(rows)


def onnx_text_plot_io(model, verbose=False, att_display=None):
    """
    Displays information about input and output types.

    :param model: ONNX graph
    :param verbose: display debugging information
    :return: str

    An ONNX graph is printed the following way:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import numpy
        from sklearn.cluster import KMeans
        from mlprodict.plotting.plotting import onnx_text_plot_io
        from mlprodict.onnx_conv import to_onnx

        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = KMeans(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_text_plot_io(onx, verbose=False)
        print(text)
    """
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
            inp.name, _get_type(inp), _get_shape(inp)))
    # initializer
    for init in model.initializer:
        rows.append("init: name=%r type=%r shape=%r" % (
            init.name, _get_type(init), _get_shape(init)))
    # outputs
    for out in model.output:
        rows.append("output: name=%r type=%r shape=%r" % (
            out.name, _get_type(out), _get_shape(out)))
    return "\n".join(rows)
