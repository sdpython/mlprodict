
=======
Xop API
=======

Most of the converting libraries uses :epkg:`onnx` to create ONNX graphs.
The API is quite verbose and that's why most of them implement a second
API wrapping the first one. They are not necessarily meant to be used
by users to create ONNX graphs as they are specialized for the training
framework they are developped for.

The API described below is similar to the one implemented in
:epkg:`sklearn-onnx` but does not depend on it. It be easily moved
to a separate package. `Xop` is the contraction of *ONNX Operators*.

.. contents::
    :local:

Short Example
=============

Let's say we need to create a graph computed the square loss between
two float tensor `X` and `Y`.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    # This line creates one class for the operator Sub and Mul.
    # It fails if the operators are misspelled.
    OnnxSub, OnnxMul = loadop('Sub', 'Mul')

    # Inputs are defined by their name as strings.
    diff = OnnxSub('X', 'Y')
    error = OnnxMul(diff, diff)

    # Then we create the ONNX graph defining 'X' and 'Y' as float.
    onx = error.to_onnx(numpy.float32, numpy.float32)

    # We check it does what it should.
    X = numpy.array([4, 5], dtype=numpy.float32)
    Y = numpy.array([4.3, 5.7], dtype=numpy.float32)

    sess = OnnxInference(onx)
    name = sess.output_names
    result = sess.run({'X': X, 'Y': Y})
    assert_almost_equal((X - Y) ** 2, result[name[0]])

    # Finally, we show the content of the graph.
    print(onnx_simple_text_plot(onx))

Visually, the model looks like the following.

.. gdot::
    :script: DOT-SECTION

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    OnnxSub, OnnxMul = loadop('Sub', 'Mul')
    diff = OnnxSub('X', 'Y')
    error = OnnxMul(diff, diff)
    onx = error.to_onnx(numpy.float32, numpy.float32)
    oinf = OnnxInference(onx, inplace=False)

    print("DOT-SECTION", oinf.to_dot())

In the following example, a string such as `'X'` refers to an input
of the graph. Every class `Onnx*` such as `OnnxSub` or `OnnxMul`
following the signature implied in ONNX specifications
(:epkg:`ONNX Operators`).
The API supports operators listed here :ref:`l-xop-api-supported-ops`.

Initializers
============

Every numpy array defined as an input of an operator
is automatically converted into an initializer.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    OnnxSub = loadop('Sub')

    # 'X' is an input, the second argument is a constant
    # stored as an initializer in the graph.
    diff = OnnxSub('X', numpy.array([1], dtype=numpy.float32))

    # Then we create the ONNX graph defining 'X' and 'Y' as float.
    onx = diff.to_onnx(numpy.float32, numpy.float32)

    # We check it does what it should.
    X = numpy.array([4, 5], dtype=numpy.float32)
    sess = OnnxInference(onx)
    name = sess.output_names
    result = sess.run({'X': X})
    assert_almost_equal(X - 1, result[name[0]])

    # Finally, we show the content of the graph.
    print(onnx_simple_text_plot(onx))

There are as many initializers as numpy arrays defined in the graph.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    OnnxSub = loadop('Sub')

    diff = OnnxSub('X', numpy.array([1], dtype=numpy.float32))
    diff2 = OnnxSub(diff, numpy.array([2], dtype=numpy.float32))
    onx = diff2.to_onnx(numpy.float32, numpy.float32)
    print(onnx_simple_text_plot(onx))

However, the conversion into onnx then applies function
:func:`onnx_optimisations
<mlprodict.onnx_tools.optim._main_onnx_optim.onnx_optimisations>`
to remove duplicated initializers. It also removes unnecessary
node such as Identity nodes or unused nodes.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    OnnxSub = loadop('Sub')

    diff = OnnxSub('X', numpy.array([1], dtype=numpy.float32))
    diff2 = OnnxSub(diff, numpy.array([1], dtype=numpy.float32))
    onx = diff2.to_onnx(numpy.float32, numpy.float32)
    print(onnx_simple_text_plot(onx))

Attributes
==========

Some operators needs attributes such as operator
:ref:`Transpose <l-xop-onnx-OnnxTranspose>`. They are
defined as named arguments.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    OnnxMatMul, OnnxTranspose = loadop('MatMul', 'Transpose')

    # Named attribute perm defines the permutation.
    result = OnnxMatMul('X', OnnxTranspose('X', perm=[1, 0]))
    onx = result.to_onnx(numpy.float32, numpy.float32)
    print(onnx_simple_text_plot(onx))

    # discrepancies?
    X = numpy.array([[4, 5]], dtype=numpy.float32)
    sess = OnnxInference(onx)
    name = sess.output_names
    result = sess.run({'X': X.copy()})
    assert_almost_equal(X @ X.T, result[name[0]])

Operator :ref:`Cast <l-xop-onnx-OnnxCast>` is used to convert
every element of an array into another type. ONNX types
and numpy types are different but the API is able to replace
one by the correspondance type.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    OnnxCast = loadop('Cast')

    result = OnnxCast('X', to=numpy.int64)
    onx = result.to_onnx(numpy.float32, numpy.int64)
    print(onnx_simple_text_plot(onx))

    # discrepancies?
    X = numpy.array([[4, 5]], dtype=numpy.float32)
    sess = OnnxInference(onx)
    name = sess.output_names
    result = sess.run({'X': X})
    assert_almost_equal(X.astype(numpy.int64), result[name[0]])

Implicit use of ONNX operators
==============================

ONNX defines standard matrix operator associated to operators
+, -, *, /, @. The API implicitely replaces them by the corresponding
ONNX operator. In the following example, operator `OnnxMatMul`
was replaced by operator `@`. The final ONNX graph looks the same.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    OnnxIdentity, OnnxTranspose = loadop('Identity', 'Transpose')

    # @ is implicity replaced by OnnxMatMul
    result = OnnxIdentity('X') @ OnnxTranspose('X', perm=[1, 0])
    onx = result.to_onnx(numpy.float32, numpy.float32)
    print(onnx_simple_text_plot(onx))

    # discrepancies?
    X = numpy.array([[4, 5]], dtype=numpy.float32)
    sess = OnnxInference(onx)
    name = sess.output_names
    result = sess.run({'X': X.copy()})
    assert_almost_equal(X @ X.T, result[name[0]])

Operator `@` only applies on class :class:`OnnxOperator
<mlprodict.npy.xop.OnnxOperator>` not on strings.
This is the base class for every class
:ref:`Identity <l-xop-onnx-OnnxIdentity>`,
:ref:`Transpose <l-xop-onnx-OnnxTranspose>`, ...
Operator :ref:`Identity <l-xop-onnx-OnnxIdentity>`
is inserted to wrap input `'X'` and enables the possibility
to use standard operations +, -, *, /, @, >, >=, ==, !=, <, <=, and, or.

Operators with multiple outputs
===============================

Operator :ref:`TopK <l-xop-onnx-OnnxTopK>` returns two results.
Accessing one of them requires the use of `[]`. The following example
extracts the two greatest elements per rows, uses the positions of
them to select the corresponding weight in another matrix,
multiply them and returns the average per row.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.npy.xop import loadop
    from mlprodict.onnxrt import OnnxInference

    OnnxReduceMean, OnnxTopK, OnnxGatherElements = loadop(
        'ReduceMean', 'TopK', 'GatherElements')

    # @ is implicity replaced by OnnxMatMul
    topk = OnnxTopK('X', numpy.array([2], dtype=numpy.int64), axis=1)
    dist = OnnxGatherElements('W', topk[1], axis=1)
    result = OnnxReduceMean(dist * topk[0], axes=[1])
    onx = result.to_onnx(numpy.float32, numpy.float32)
    print(onnx_simple_text_plot(onx))

    # discrepancies?
    X = numpy.array([[4, 5, 6], [7, 0, 1]], dtype=numpy.float32)
    W = numpy.array([[1, 0.5, 0.6], [0.5, 0.2, 0.3]], dtype=numpy.float32)
    sess = OnnxInference(onx)
    name = sess.output_names[0]
    result = sess.run({'X': X, 'W': W})
    print('\nResults:')
    print(result[name])

Sub Estimators
==============

It is a common need to insert an ONNX graph into another one.
It is not a simple merge, there are operations before and after
and the ONNX graph may have been produced by another library.
That's the purpose of class :class:`OnnxSubOnnx
<mlprodict.npy.xop_convert.OnnxSubOnnx>`.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.npy.xop_convert import OnnxSubOnnx
    from mlprodict.npy.xop import loadop
    from mlprodict.onnxrt import OnnxInference

    OnnxIdentity = loadop('Identity')

    X = numpy.array([[-1.5, -0.5, 0.5, 1.5]], dtype=numpy.float32)

    # Let's create a first ONNX graph which implements
    # a Relu function.
    vx = OnnxIdentity('X')
    sign = vx > numpy.array([0], dtype=numpy.float32)
    sign_float = sign.astype(numpy.float32)
    relu = vx * sign_float
    print('-- Relu graph')
    onx_relu = relu.to_onnx(numpy.float32, numpy.float32)

    print("-- Relu results")
    print(onnx_simple_text_plot(onx_relu))
    sess = OnnxInference(onx_relu)
    name = sess.output_names[0]
    result = sess.run({'X': X})
    print('-- Results:')
    print(result[name])

    # Then the second graph including the first one.
    x_1 = OnnxIdentity('X') + numpy.array([1], dtype=numpy.float32)

    # Class OnnxSubOnnx takes a graph as input and applies it on the
    # given inputs.
    result = OnnxSubOnnx(onx_relu, x_1)

    onx = result.to_onnx(numpy.float32, numpy.float32)
    print('-- Whole graph')
    print(onnx_simple_text_plot(onx))

    # Expected results?
    sess = OnnxInference(onx)
    name = sess.output_names[0]
    result = sess.run({'X': X})
    print('-- Whole results:')
    print(result[name])

This mechanism is used to plug any model from :epkg:`scikit-learn`
converted into ONNX in a bigger graph. Next example averages
the probabilities of two classifiers for a binary classification.
That's the purpose of class :class:`OnnxSubEstimator
<mlprodict.npy.xop_convert.OnnxSubEstimator>`. The class automatically
calls the appropriate converter, :epkg:`sklearn-onnx` for
:epkg:`scikit-learn` models.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.npy.xop_convert import OnnxSubEstimator
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    # machine learning part
    X, y = make_classification(1000, n_classes=2, n_features=5, n_redundant=0)
    X = X.astype(numpy.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # we train two models not on the same machine
    lr1 = LogisticRegression().fit(X_train[:, :2], y_train)
    lr2 = LogisticRegression().fit(X_train[:, 2:], y_train)

    # score?
    p1 = lr1.predict_proba(X_test[:, :2])
    print("score1", roc_auc_score(y_test, p1[:, 1]))
    p2 = lr2.predict_proba(X_test[:, 2:])
    print("score2", roc_auc_score(y_test, p2[:, 1]))

    # OnnxGraph

    OnnxIdentity, OnnxGather = loadop('Identity', 'Gather')

    x1 = OnnxGather('X', numpy.array([0, 1], dtype=numpy.int64), axis=1)
    x2 = OnnxGather('X', numpy.array([2, 3, 4], dtype=numpy.int64), axis=1)

    # Class OnnxSubEstimator inserts the model into the ONNX graph.
    p1 = OnnxSubEstimator(lr1, x1, initial_types=X_train[:, :2])
    p2 = OnnxSubEstimator(lr2, x2, initial_types=X_train[:, 2:])
    result = ((OnnxIdentity(p1[1]) + OnnxIdentity(p2[1])) /
        numpy.array([2], dtype=numpy.float32))

    # Then the second graph including the first one.
    onx = result.to_onnx(numpy.float32, numpy.float32)
    print('-- Whole graph')
    print(onnx_simple_text_plot(onx))

    # Expected results?
    sess = OnnxInference(onx)
    name = sess.output_names[0]
    result = sess.run({'X': X_test})[name]

    print("score3", roc_auc_score(y_test, result[:, 1]))

Subgraphs
=========

Inputs, outputs
===============

Opsets
======
