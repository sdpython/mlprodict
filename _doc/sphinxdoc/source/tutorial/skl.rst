From scikit-learn to ONNX
=========================

Function `skl2onnx.to_onnx <http://www.xavierdupre.fr/app/sklearn-onnx/helpsphinx/
api_summary.html#skl2onnx.to_onnx>`_ is the
main entrypoint to convert a *scikit-learn* pipeline into ONNX.
The same function was extended in this package into
:func:`to_onnx <mlprodict.onnx_conv.convert.to_onnx>` to handle
dataframes, an extended list of supported converters, scorers.
It works exactly the same:

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from mlprodict.onnx_conv import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X = iris.data.astype(numpy.float32)
    X_train, X_test = train_test_split(X)
    clr = KMeans(n_clusters=3)
    clr.fit(X_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32),
                        target_opset=12)

    oinf = OnnxInference(model_def, runtime='python')
    print(oinf.run({'X': X_test[:5]}))

This new version extends the conversion to scorers through
:func:`convert_scorer <mlprodict.onnx_conv.convert.convert_scorer>`.
