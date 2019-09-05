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

        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import load_iris
        from mlprodict.onnxrt.optim.onnx_helper import onnx_statistics

        iris = load_iris()
        X = iris.data
        y = iris.target
        lr = LogisticRegression()
        lr.fit(X, y)

        import pprint
        pprint.pprint(lr, onnx_statistics(lr))

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        from mlprodict.onnxrt.optim.sklearn_helper import inspect_sklearn_model

        iris = load_iris()
        X = iris.data
        y = iris.target
        rf = RandomForestClassifier()
        rf.fit(X, y)

        import pprint
        pprint.pprint(rf, onnx_statistics(rf))
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
