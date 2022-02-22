
.. _l-onnx-tutorial:

Execute ONNX graphs
===================

This package implements a python runtime for ONNX
in class :class:`OnnxInference <mlprodict.onnxrt.OnnxInference>`.
It does not depend on :epkg:`scikit-learn`, only :epkg:`numpy`
and this module. However, this module was not really developped to
get the fastest python runtime but mostly to easily develop converters.

.. contents::
    :local:

.. _l-onnx-python-runtime:

Python Runtime for ONNX
+++++++++++++++++++++++

Class :class:`OnnxInference <mlprodict.onnxrt.onnx_inference.OnnxInference>`
implements a python runtime for a subset of ONNX operators needed
to convert many :epkg:`scikit-learn` models.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from skl2onnx import to_onnx
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

It is usually useful to get information on intermediate results
in the graph itself to understand where the discrepencies
begin.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from skl2onnx import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X = iris.data.astype(numpy.float32)
    X_train, X_test = train_test_split(X)
    clr = KMeans(n_clusters=3)
    clr.fit(X_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32),
                        target_opset=12)

    oinf = OnnxInference(model_def, runtime='python')
    print(oinf.run({'X': X_test[:5]}, verbose=1, fLOG=print))

.. index:: intermediate results, verbosity

The verbosity can be increased.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from skl2onnx import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X = iris.data.astype(numpy.float32)
    X_train, X_test = train_test_split(X)
    clr = KMeans(n_clusters=3)
    clr.fit(X_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32),
                        target_opset=12)

    oinf = OnnxInference(model_def, runtime='python')
    print(oinf.run({'X': X_test[:5]}, verbose=3, fLOG=print))

Other runtimes with OnnxInference
+++++++++++++++++++++++++++++++++

:class:`OnnxInference <mlprodict.onnxrt.onnx_inference.OnnxInference>`
can also call :epkg:`onnxruntime` to compute the predictions by using
``runtime='onnxruntime1'``.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from skl2onnx import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X = iris.data.astype(numpy.float32)
    X_train, X_test = train_test_split(X)
    clr = KMeans(n_clusters=3)
    clr.fit(X_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32),
                        target_opset=12)

    oinf = OnnxInference(model_def, runtime='onnxruntime1')
    print(oinf.run({'X': X_test[:5]}))

Intermediate cannot be seen but the class may decompose
the ONNX graph into smaller graphs, one per operator,
to look into intermediate results.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from skl2onnx import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X = iris.data.astype(numpy.float32)
    X_train, X_test = train_test_split(X)
    clr = KMeans(n_clusters=3)
    clr.fit(X_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32),
                        target_opset=12)

    oinf = OnnxInference(model_def, runtime='onnxruntime2')
    print(oinf.run({'X': X_test[:5]}, verbose=1, fLOG=print))

Finally, a last runtime `'python_compiled'` converts some
part of the class :class:`OnnxInference
<mlprodict.onnxrt.onnx_inference.OnnxInference>`
into python code then dynamically compiled.
As a consequence, interdiate results cannot be seen anymore.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from skl2onnx import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X = iris.data.astype(numpy.float32)
    X_train, X_test = train_test_split(X)
    clr = KMeans(n_clusters=3)
    clr.fit(X_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32),
                        target_opset=12)

    oinf = OnnxInference(model_def, runtime='python_compiled')
    print(oinf.run({'X': X_test[:5]}))
