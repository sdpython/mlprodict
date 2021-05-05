"""
@file
@brief Statistics on :epkg:`ONNX` models.
"""
from collections import Counter
from onnx.helper import make_graph
from onnx import ValueInfoProto
from skl2onnx.common._topology import Variable
from ._onnx_optimisation_common import _apply_optimisation_on_graph
from .onnx_optimisation import onnx_remove_node


def onnx_statistics(onnx_model, recursive=True, optim=True):
    """
    Computes statistics on :epkg:`ONNX` models,
    extracts informations about the model such as
    the number of nodes.

    @param      onnx_model      onnx model
    @param      recursive       looks into subgraphs
    @param      optim           adds statistics because of optimisation
    @return                     dictionary

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import pprint
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        from mlprodict.onnxrt.optim.onnx_helper import onnx_statistics
        from mlprodict.onnx_conv import to_onnx

        iris = load_iris()
        X = iris.data
        y = iris.target
        lr = LogisticRegression()
        lr.fit(X, y)
        onx = to_onnx(lr, X[:1])
        pprint.pprint((lr, onnx_statistics(onx)))

        iris = load_iris()
        X = iris.data
        y = iris.target
        rf = RandomForestClassifier()
        rf.fit(X, y)
        onx = to_onnx(rf, X[:1], target_opset=12)
        pprint.pprint((rf, onnx_statistics(onx)))
    """
    atts = ['doc_string', 'ir_version', 'metadata_props', 'domain',
            'model_version', 'producer_name', 'producer_version']

    def update(sts, st):
        for k, v in st.items():
            if k in ['size'] or k in atts:
                continue  # pragma: no cover
            if k in sts:
                sts[k] += v
            else:
                sts[k] = v

    if hasattr(onnx_model, 'graph'):
        content = onnx_model.SerializeToString()
        nnodes = len(onnx_model.graph.node)
        ninits = len(onnx_model.graph.initializer)
        stats = {'size': len(content), 'nnodes': nnodes, 'ninits': ninits}
        for a in atts:
            v = getattr(onnx_model, a)
            if isinstance(v, str):
                li = None
            else:
                try:
                    li = list(v)
                except TypeError:
                    li = None
            if li is not None and len(li) == 0:
                continue
            stats[a] = v

        for opi in onnx_model.opset_import:
            stats[opi.domain] = opi.version

        graph = onnx_model.graph
    elif not hasattr(onnx_model, 'node'):  # pragma: no cover
        # We're in a node.
        stats = {'nnodes': 1}
        if hasattr(onnx_model, 'attribute') and onnx_model.attribute:
            for att in onnx_model.attribute:
                if att.name == 'body':
                    st = onnx_statistics(att.g)
                    update(stats, st)
        return stats
    else:
        graph = onnx_model
        nnodes = len(graph.node)
        stats = {'nnodes': nnodes}

    # Number of identities
    counts = Counter(map(lambda obj: obj.op_type, graph.node))
    for op in ['Cast', 'Identity', 'ZipMap', 'Reshape']:
        if op in counts:
            stats['op_' + op] = counts[op]

    # Recursive
    if recursive:
        for node in graph.node:
            if not hasattr(node, 'attribute'):
                continue  # pragma: no cover
            for att in node.attribute:
                if att.name != 'body':
                    continue
                substats = onnx_statistics(att.g, recursive=True, optim=False)
                update(stats, {'subgraphs': 1})
                update(stats, substats)

    # optimisation: remove_identity nodes
    if optim:
        new_model = onnx_remove_node(
            onnx_model, recursive=recursive)
        st = onnx_statistics(new_model, recursive=recursive, optim=False)
        for key in ["op_Identity", "subgraphs", "size",
                    "nnodes", "ninits"]:
            if key in st:
                stats[key + "_optim"] = st[key]
    return stats


def change_input_first_dimension(onnx_model, N=None, debug_info=None):
    """
    Some models are converted under the assumption
    batch prediction is not necessary. This function
    changes the first dimension of an ONNX graph.

    @param      onnx_model      model :epkg:`onnx`
    @param      N               new first dimension,
                                None to avoid changing it,
                                0 to fix an undefined
                                first dimension
    @param      debug_info      unused
    @return                     modified model onnx
    """
    def _make_value_info(variable):
        value_info = ValueInfoProto()
        value_info.name = variable.full_name
        value_info.type.CopyFrom(  # pylint: disable=E1101
            variable.type.to_onnx_type())  # pylint: disable=E1101
        if variable.type.doc_string:  # pylint: disable=E0611
            value_info.doc_string = variable.type.doc_string  # pragma: no cover
        return value_info

    if hasattr(onnx_model, 'graph'):
        return _apply_optimisation_on_graph(
            change_input_first_dimension, onnx_model, N=N)

    graph = onnx_model

    nodes = graph.node
    inputs = [Variable.from_pb(input) for input in onnx_model.input]
    outputs = onnx_model.output

    if N <= 0:
        N = None
    for input in inputs:
        input.type.shape[0] = N
    inputs = [_make_value_info(v) for v in inputs]

    graph = make_graph(nodes, onnx_model.name,
                       inputs, outputs, onnx_model.initializer)

    graph.value_info.extend(onnx_model.value_info)  # pylint: disable=E1101
    return graph
