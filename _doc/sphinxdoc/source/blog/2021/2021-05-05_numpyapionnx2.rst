
.. blogpost::
    :title: Numpy API for ONNX and scikit-learn (part II)
    :keywords: ONNX, API, numpy, scikit-learn
    :date: 2021-05-05
    :categories: API
    :lid: blog-onnx-api-part2

    This follows blog post :ref:`Numpy API for ONNX and scikit-learn (part I)
    <blog-onnx-api-part1>`. It demonstrated how to insert a custom
    function in a pipeline and still be able to convert that pipeline
    into ONNX. This blog post shows how to implement a custom transformer.

    This time, we need to implement method not a function but the method
    `transform` of a custom transformer. The design is the same
    and relies on a decorator before the class declaration.
    In the following example, a method `onnx_transform`
    implements the method transform with the API mentioned
    in the first part: :ref:`f-numpyonnximpl`.
    The decorator `onnxsklearn_class` detects that the decorated class
    is a transform. It then assumes that method `onnx_transform`
    contains the ONNX implementation of method `transform`.
    The decorator adds an implementation for method `transform`.
    It behaves like the custom function described in part I.
    Once called, this method will detects the input type,
    generates the ONNX graph if not available and executes it
    with a runtimme. That explains why the first call is much slower.

    .. runpython::
        :showcode:
        :process:

        import numpy
        from pandas import DataFrame
        from sklearn.base import TransformerMixin, BaseEstimator
        from sklearn.decomposition import PCA
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_classification
        from mlprodict.npy import onnxsklearn_class
        from mlprodict.onnx_conv import to_onnx
        from mlprodict.plotting.text_plot import onnx_simple_text_plot
        import mlprodict.npy.numpy_onnx_impl as nxnp
        import mlprodict.npy.numpy_onnx_impl_skl as nxnpskl

        X, y = make_classification(200, n_classes=2, n_features=2, n_informative=2,
                                n_redundant=0, n_clusters_per_class=2, hypercube=False)

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        @onnxsklearn_class("onnx_transform", op_version=14)  # opset=13, 14, ...
        class DecorrelateTransformerOnnx(TransformerMixin, BaseEstimator):
            def __init__(self, alpha=0.):
                BaseEstimator.__init__(self)
                TransformerMixin.__init__(self)
                self.alpha = alpha

            def fit(self, X, y=None, sample_weights=None):
                self.pca_ = PCA(X.shape[1])  # pylint: disable=W0201
                self.pca_.fit(X)
                return self

            def onnx_transform(self, X):
                if X.dtype is None:
                    raise AssertionError("X.dtype cannot be None.")
                mean = self.pca_.mean_.astype(X.dtype)
                cmp = self.pca_.components_.T.astype(X.dtype)
                return (X - mean) @ cmp

        model = DecorrelateTransformerOnnx()
        model.fit(X_train)
        print(model.transform(X_test[:5]))

        onx = to_onnx(model, X_test[:5], target_opset=14)  # opset=13, 14, ...
        print()
        print(onnx_simple_text_plot(onx))
        print()
        print(onx)

    The tutorial :ref:`l-numpy-api-for-onnx` extends this example
    to regressors or classifiers. It also mentions a couple of frequent
    errors that may appear along the way.
