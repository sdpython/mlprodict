
.. blogpost::
    :title: Numpy API for ONNX and scikit-learn (part I)
    :keywords: ONNX, API, numpy, scikit-learn
    :date: 2021-05-05
    :categories: API
    :lid: blog-onnx-api-part1

    :epkg:`sklearn-onnx` converts most of the pipelines including
    numerical preprocessing or predictors but it fails whenever
    custom code is involved. That covers the use of `FunctionTransformer
    <https://scikit-learn.org/stable/modules/generated/
    sklearn.preprocessing.FunctionTransformer.html>`_ or a new model
    inheriting from `BaseEstimator <https://scikit-learn.org/stable/
    modules/generated/sklearn.base.BaseEstimator.html>`_. To be successful,
    the conversion needs a way to convert the custom code into ONNX.
    The proposed solution here is bypass that complex steps
    (rewrite a python function with ONNX operators) by directly writing
    the custom code with ONNX operators. However, even though most of
    the operator are close to :epkg:`numpy` functions, they are not
    the same. To avoid spending time looking at them, many :epkg:`numpy`
    functions were implementing with ONNX operators. The custom function
    or predictor can then just be implemented with this API to build
    a unique ONNX graph executed with a runtime.

    Next sections takes some examples from
    :ref:`l-numpy-api-for-onnx`.

    **numpy API for ONNX**

    Let's an example with a `FunctionTransformer
    <https://scikit-learn.org/stable/modules/generated/
    sklearn.preprocessing.FunctionTransformer.html>`_.

    The mechanism is similar to what :epkg:`pytorch` or :epkg:`tensorflow`
    put in place: write a graph assuming every node processes a variable.
    Then the user instantiates a variable and executes the graph.
    It works the same with ONNX. The following snippet implement the
    function :math:`log(1 + x)`.

    ::

        import numpy as np
        import mlprodict.npy.numpy_onnx_impl as npnx

        def onnx_log_1(x):
            return npnx.log(x + np.float32(1))

    The list of implemented function is :ref:`f-numpyonnximpl`.
    ONNX is strongly typed so we need to specified them with annotations.

    ::

        from typing import Any
        import numpy as np
        from mlprodict.npy import NDArray
        import mlprodict.npy.numpy_onnx_impl as npnx

        def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[(None, None), np.float32]:
            return npnx.log(x + np.float32(1))

    And finally, this function does not run on a numpy array as every
    function expects a variable (see :class:`OnnxVariable
    <mlprodict.npy.onnx_variable.OnnxVariable>`) to define an ONNX graph
    which can be executed with a runtime. That's the purpose of the decorator
    `onnxnumpy_default`.

    .. runpython::
        :showcode:
        :process:

        from typing import Any
        import numpy as np
        from mlprodict.npy import onnxnumpy_default, NDArray
        import mlprodict.npy.numpy_onnx_impl as npnx

        @onnxnumpy_default
        def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[(None, None), np.float32]:
            return npnx.log(x + np.float32(1))

        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        print(onnx_log_1(x))
        print(type(onnx_log_1))

    `onnx_log_1` is not a function but an instance
    of a class which defines operator `__call__` and that class
    has a hold on the ONNX graph and all the necessary information
    to have :epkg:`sklearn-onnx` convert any pipeline using it after
    a new converter for `FunctionTransformer
    <https://scikit-learn.org/stable/modules/generated/
    sklearn.preprocessing.FunctionTransformer.html>`_ is registered
    to handle this API.

    The ONNX graph is created when the function is called for the
    first time and loaded by the runtime. That explains why the first
    call is much slower and all the other call.

    ::

        from mlprodict.onnx_conv import register_rewritten_operators
        register_rewritten_operators()

    **The complete example:**

    .. runpython::
        :showcode:
        :process:

        from typing import Any
        import numpy as np
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        from sklearn.linear_model import LogisticRegression
        import mlprodict.npy.numpy_onnx_impl as npnx
        from mlprodict.npy import onnxnumpy_default, NDArray
        from mlprodict.onnxrt import OnnxInference

        from skl2onnx import to_onnx
        from mlprodict.onnx_conv import register_rewritten_operators
        register_rewritten_operators()

        @onnxnumpy_default
        def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[(None, None), np.float32]:
            return npnx.log(x + np.float32(1))

        data = load_iris()
        X, y = data.data.astype(np.float32), data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pipe = make_pipeline(
            FunctionTransformer(onnx_log_1),
            StandardScaler(),
            LogisticRegression())
        pipe.fit(X_train, y_train)
        print(pipe.predict_proba(X_test[:2]))

        onx = to_onnx(pipe, X_train[:1],
                      options={LogisticRegression: {'zipmap': False}})
        oinf = OnnxInference(onx)
        print(oinf.run({'X': X_test[:2]})['probabilities'])

    The decorator has parameter to change the way the function
    is converted or executed. ONNX has different version or opset,
    it is possible to target a specific opset. The ONNX graph must
    be executed with a runtime, this one or :epkg:`onnxruntime`.
    This can be defined too. The function is strongly typed but it is
    possible to have an implementation which supports multiple types.
    An ONNX graph will be created for every distinct type,
    like a template in C++.
    See :ref:`l-numpy-api-for-onnx` for more information.

    Next: :ref:`Numpy API for ONNX and scikit-learn (part II) <blog-onnx-api-part2>`.
