"""
@file
@brief Text representations of graphs.
"""
from collections import OrderedDict
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

        import numpy
        from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxSub
        from mlprodict.onnx_conv import to_onnx
        from mlprodict.tools import get_opset_number_from_onnx
        from mlprodict.plotting.plotting import onnx_text_plot

        idi = numpy.identity(2)
        opv = get_opset_number_from_onnx()
        A = OnnxAdd('X', idi, op_version=opv)
        B = OnnxSub(A, 'W', output_names=['Y'], op_version=opv)
        onx = B.to_onnx({'X': idi.astype(numpy.float32),
                         'W': idi.astype(numpy.float32)})
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
        if r == b'BRANCH_LT':
            return '<'
        if r == b'BRANCH_GEQ':
            return '>='
        if r == b'BRANCH_GT':
            return '>'
        if r == b'BRANCH_EQ':
            return '=='
        if r == b'BRANCH_NEQ':
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
