"""
@file
@brief Statistics on :epkg:`ONNX` models.
"""
from collections import Counter
from .onnx_optimisation import onnx_remove_node


def onnx_statistics(onnx_model, recursive=True, optim=True):
    """
    Computes statistics on :epkg:`ONNX` models.

    @param      onnx_model      onnx model
    @param      recursive       looks into subgraphs
    @param      optim           adds statistics because of optimisation
    @return                     dictionary

    .. runpython::
        :showcode:

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
        onx = to_onnx(rf, X[:1])
        pprint.pprint((rf, onnx_statistics(onx)))
    """
    atts = ['doc_string', 'ir_version', 'metadata_props', 'domain',
            'model_version', 'producer_name', 'producer_version']

    def update(sts, st):
        for k, v in st.items():
            if k in ['size'] or k in atts:
                continue
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
    elif not hasattr(onnx_model, 'node'):
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
                continue
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
