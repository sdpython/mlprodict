"""
@file
@brief Statistics on :epkg:`ONNX` models.
"""


def onnx_statistics(onnx_model):
    """
    Computes statistics on :epkg:`ONNX` models.

    @param      onnx_model      onnx model
    @return                     dictionary

    .. runpython::
        :showcode:

        import pprint
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        from mlprodict.onnxrt.optim.onnx_helper import onnx_statistics
        from mlprodict.onnxrt import to_onnx

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
    content = onnx_model.SerializeToString()
    nnodes = len(onnx_model.graph.node)
    stats = {'size': len(content), 'nnodes': nnodes}
    atts = ['doc_string', 'ir_version', 'metadata_props', 'domain',
            'model_version', 'producer_name', 'producer_version']
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
    return stats
