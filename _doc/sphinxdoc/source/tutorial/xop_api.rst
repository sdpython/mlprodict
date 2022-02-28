
.. _l-xop-api:

=======
Xop API
=======

Most of the converting libraries uses :epkg:`onnx` to create ONNX graphs.
The API is quite verbose and that is why most of them implement a second
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
That is the purpose of class :class:`OnnxSubOnnx
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

    print("\n-- Relu results")
    print(onnx_simple_text_plot(onx_relu))
    sess = OnnxInference(onx_relu)
    name = sess.output_names[0]
    result = sess.run({'X': X})
    print('\n-- Results:')
    print(result[name])

    # Then the second graph including the first one.
    x_1 = OnnxIdentity('X') + numpy.array([1], dtype=numpy.float32)

    # Class OnnxSubOnnx takes a graph as input and applies it on the
    # given inputs.
    result = OnnxSubOnnx(onx_relu, x_1)

    onx = result.to_onnx(numpy.float32, numpy.float32)
    print('\n-- Whole graph')
    print(onnx_simple_text_plot(onx))

    # Expected results?
    sess = OnnxInference(onx)
    name = sess.output_names[0]
    result = sess.run({'X': X})
    print('\n-- Whole results:')
    print(result[name])

This mechanism is used to plug any model from :epkg:`scikit-learn`
converted into ONNX in a bigger graph. Next example averages
the probabilities of two classifiers for a binary classification.
That is the purpose of class :class:`OnnxSubEstimator
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
    print('\n-- Whole graph')
    print(onnx_simple_text_plot(onx))

    # Expected results?
    sess = OnnxInference(onx)
    name = sess.output_names[0]
    result = sess.run({'X': X_test})[name]

    print("\nscore3", roc_auc_score(y_test, result[:, 1]))

.. gdot::
    :script: DOT-SECTION

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

    X, y = make_classification(1000, n_classes=2, n_features=5, n_redundant=0)
    X = X.astype(numpy.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lr1 = LogisticRegression().fit(X_train[:, :2], y_train)
    lr2 = LogisticRegression().fit(X_train[:, 2:], y_train)

    p1 = lr1.predict_proba(X_test[:, :2])
    print("score1", roc_auc_score(y_test, p1[:, 1]))
    p2 = lr2.predict_proba(X_test[:, 2:])
    print("score2", roc_auc_score(y_test, p2[:, 1]))

    OnnxIdentity, OnnxGather = loadop('Identity', 'Gather')

    x1 = OnnxGather('X', numpy.array([0, 1], dtype=numpy.int64), axis=1)
    x2 = OnnxGather('X', numpy.array([2, 3, 4], dtype=numpy.int64), axis=1)

    # Class OnnxSubEstimator inserts the model into the ONNX graph.
    p1 = OnnxSubEstimator(lr1, x1, initial_types=X_train[:, :2])
    p2 = OnnxSubEstimator(lr2, x2, initial_types=X_train[:, 2:])
    result = ((OnnxIdentity(p1[1]) + OnnxIdentity(p2[1])) /
        numpy.array([2], dtype=numpy.float32))

    onx = result.to_onnx(numpy.float32, numpy.float32)
    oinf = OnnxInference(onx, inplace=False)

    print("DOT-SECTION", oinf.to_dot())

Inputs, outputs
===============

The following code does not specify on which type it applies,
float32, float64, it could be a tensor of any of numerical type.

.. runpython::
    :showcode:

    from mlprodict.npy.xop import loadop

    OnnxSub, OnnxMul = loadop('Sub', 'Mul')

    diff = OnnxSub('X', 'Y')
    error = OnnxMul(diff, diff)
    print(error)

That is why this information must be specified when it is being
converted into ONNX. That explains why method :meth:`to_onnx
<mlprodict.npy.xop.OnnxOperator.to_onnx>` needs more information
to convert the object into ONNX: `to_onnx(<input type>, <output type>)`.

.. runpython::
    :showcode:

    import numpy
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.npy.xop import loadop

    OnnxSub, OnnxMul = loadop('Sub', 'Mul')

    diff = OnnxSub('X', 'Y')
    error = OnnxMul(diff, diff)

    # First numpy.float32 is for the input.
    # Second numpy.float32 is for the output.
    onx = error.to_onnx(numpy.float32, numpy.float32)
    print(onnx_simple_text_plot(onx))

Wrong types are possible however the runtime executing the graph
may raise an exception telling the graph cannot be executed.

Optional output type
++++++++++++++++++++

Most of the time the output type can be guessed based on the signature
of every operator involved in the graph. Second argument, `output_type`,
is optional.

.. runpython::
    :showcode:

    import numpy
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.npy.xop import loadop

    OnnxSub, OnnxMul = loadop('Sub', 'Mul')

    diff = OnnxSub('X', 'Y')
    error = OnnxMul(diff, diff)
    onx = error.to_onnx(numpy.float32)
    print(onnx_simple_text_plot(onx))

Multiple inputs and multiple types
++++++++++++++++++++++++++++++++++

Previous syntax assumes all inputs or outputs share the same type.
That is usually the case but not always. The order of inputs
is not very clear and that explains why the different types
are specifed using a dictionary using name as keys.

.. runpython::
    :showcode:

    import numpy
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.npy.xop_variable import Variable
    from mlprodict.npy.xop import loadop

    OnnxMul, OnnxReshape, OnnxReduceSum = loadop(
        'Mul', 'Reshape', 'ReduceSum')

    diff = OnnxReshape('X', 'Y')
    diff2 = OnnxMul(diff, diff)
    sumd = OnnxReduceSum(diff2, numpy.array([1], dtype=numpy.int64))
    onx = sumd.to_onnx({'X': numpy.float32, 'Y': numpy.int64},
                       numpy.float32)
    print(onnx_simple_text_plot(onx))

Specifying output types is more tricky. Types must still be specified
by names but output names are unknown. They are decided when the conversion
happens unless the user wants them to be named as his wished. That is where
argument *output_names* takes place in the story. It forces method *to_onnx*
to keep the chosen names when the model is converting into ONNX and
then we can be sure to give the proper type to the proper output.
The two ouputs are coming from two different objects, the conversion
is started by calling `to_onnx` from one and the other one is added
in argument `other_outputs`.

.. runpython::
    :showcode:

    import numpy
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.npy.xop import loadop

    OnnxMul, OnnxReshape, OnnxReduceSum, OnnxShape = loadop(
        'Mul', 'Reshape', 'ReduceSum', 'Shape')

    diff = OnnxReshape('X', 'Y')
    diff2 = OnnxMul(diff, diff)
    sumd = OnnxReduceSum(diff2, numpy.array([1], dtype=numpy.int64),
                         output_names=['Z'])
    shape = OnnxShape(sumd, output_names=['S'])
    onx = sumd.to_onnx({'X': numpy.float32, 'Y': numpy.int64},
                       {'Z': numpy.float32, 'S': numpy.int64},
                       other_outputs=[shape])
    print(onnx_simple_text_plot(onx))

Runtime for ONNX are usually better when inputs and output shapes
are known or at least some part of it. That can be done the following way.
It needs to be done through a list of :class:`Variable
<mlprodict.npy.xop_variable.Variable>`.

.. runpython::
    :showcode:

    import numpy
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.npy.xop_variable import Variable
    from mlprodict.npy.xop import loadop

    OnnxMul, OnnxReshape, OnnxReduceSum, OnnxShape = loadop(
        'Mul', 'Reshape', 'ReduceSum', 'Shape')

    diff = OnnxReshape('X', 'Y')
    diff2 = OnnxMul(diff, diff)
    sumd = OnnxReduceSum(diff2, numpy.array([1], dtype=numpy.int64),
                         output_names=['Z'])
    shape = OnnxShape(sumd, output_names=['S'])
    onx = sumd.to_onnx(
        [Variable('X', numpy.float32, [None, 2]),
         Variable('Y', numpy.int64, [2])],
        [Variable('Z', numpy.float32, [None, 1]),
         Variable('S', numpy.int64, [2])],
        other_outputs=[shape])
    print(onnx_simple_text_plot(onx))

Opsets
======

ONNX is versioned. The assumption is every old ONNX graph must remain
valid even if new verions of the language were released. By default,
the latest supported version is used. You first have the latest version
installed:

.. runpython::
    :showcode:

    from onnx.defs import onnx_opset_version
    print("onnx_opset_version() ->", onnx_opset_version())

But the library does not always support the latest version right away.
That is the default opset if none is given.

.. runpython::
    :showcode:

    import pprint
    from mlprodict import __max_supported_opset__, __max_supported_opsets__
    print(__max_supported_opset__)
    pprint.pprint(__max_supported_opsets__)

Following example shows how to force the opset to 12 instead of the
default version. It must be specified in two places, in every operator,
and when calling `to_onnx` with argument `target_opset`.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    opset = 12
    OnnxSub, OnnxMul = loadop('Sub', 'Mul')
    diff = OnnxSub('X', 'Y', op_version=opset)
    error = OnnxMul(diff, diff, op_version=opset)
    onx = error.to_onnx(numpy.float32, numpy.float32,
                        target_opset=opset)
    print(onnx_simple_text_plot(onx))

It can be also done by using the specific class corresponding to
the most recent version below the considered opset.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    opset = 12
    OnnxSub_7, OnnxMul_7 = loadop('Sub_7', 'Mul_7')
    diff = OnnxSub_7('X', 'Y')
    error = OnnxMul_7(diff, diff)
    onx = error.to_onnx(numpy.float32, numpy.float32,
                        target_opset=opset)
    print(onnx_simple_text_plot(onx))

There is one unique opset per domain. The opsets associated to
the other domains can be specified as a dictionary.

.. runpython::
    :showcode:

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop import loadop

    opset = 12
    OnnxSub_7, OnnxMul_7 = loadop('Sub_7', 'Mul_7')
    diff = OnnxSub_7('X', 'Y')
    error = OnnxMul_7(diff, diff)
    onx = error.to_onnx(numpy.float32, numpy.float32,
                        target_opset={'': opset, 'ai.onnx.ml': 1})
    print(onnx_simple_text_plot(onx))

Usually, the code written with one opset is likely to run the same way
with the next one. However, the signature of an operator may change,
an attribute may become an input. The code has to be different according
to the opset, see for example function :func:`OnnxSqueezeApi11
<mlprodict.npy.xop_opset.OnnxSqueezeApi11>`.

Subgraphs
=========

Three operators hold graph attributes or subgraph:
:class:`If <class mlprodict.npy.xop_auto_import_.OnnxIf>`,
:class:`Loop <class mlprodict.npy.xop_auto_import_.OnnxLoop>`,
:class:`Scan <class mlprodict.npy.xop_auto_import_.OnnxScan>`.
The first one executes one graph or another based on one condition.
The two others ones run loops. Those operators are not so easy
to deal with. Unittests may provide more examples
`test_xop.py
<https://github.com/sdpython/mlprodict/blob/master/_unittests/ut_npy/test_xop.py>`_.

.. runpython::
    :showcode:

    import numpy
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop_variable import Variable
    from mlprodict.npy.xop import loadop

    (OnnxSub, OnnxIdentity, OnnxReduceSumSquare, OnnxScan,
     OnnxAdd) = loadop('Sub', 'Identity',
                       'ReduceSumSquare', 'Scan', 'Add')

    # Building of the subgraph.
    diff = OnnxSub('next_in', 'next')
    id_next = OnnxIdentity('next_in', output_names=['next_out'])
    flat = OnnxReduceSumSquare(
        diff, axes=[1], output_names=['scan_out'], keepdims=0)
    scan_body = id_next.to_onnx(
        [Variable('next_in', numpy.float32, (None, None)),
         Variable('next', numpy.float32, (None, ))],
        outputs=[Variable('next_out', numpy.float32, (None, None)),
                 Variable('scan_out', numpy.float32, (None, ))],
        other_outputs=[flat])
    output_names = [o.name for o in scan_body.graph.output]

    cop = OnnxAdd('input', 'input')

    # Subgraph as a graph attribute.
    node = OnnxScan(cop, cop, output_names=['S1', 'S2'],
                    num_scan_inputs=1,
                    body=(scan_body.graph, [id_next, flat]))

    cop2 = OnnxIdentity(node[1], output_names=['cdist'])

    model_def = cop2.to_onnx(numpy.float32, numpy.float32)

    x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
        numpy.float32).reshape((3, 2))
    sess = OnnxInference(model_def)
    res = sess.run({'input': x})
    print(res)

    print("\n-- Graph:")
    print(onnx_simple_text_plot(model_def, recursive=True))

And visually:

.. gdot::
    :script: DOT-SECTION

    import numpy
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.npy.xop_variable import Variable
    from mlprodict.npy.xop import loadop

    (OnnxSub, OnnxIdentity, OnnxReduceSumSquare, OnnxScan,
     OnnxAdd) = loadop('Sub', 'Identity',
                       'ReduceSumSquare', 'Scan', 'Add')

    # Building of the subgraph.
    diff = OnnxSub('next_in', 'next')
    id_next = OnnxIdentity('next_in', output_names=['next_out'])
    flat = OnnxReduceSumSquare(
        diff, axes=[1], output_names=['scan_out'], keepdims=0)
    scan_body = id_next.to_onnx(
        [Variable('next_in', numpy.float32, (None, None)),
         Variable('next', numpy.float32, (None, ))],
        outputs=[Variable('next_out', numpy.float32, (None, None)),
                 Variable('scan_out', numpy.float32, (None, ))],
        other_outputs=[flat])
    output_names = [o.name for o in scan_body.graph.output]

    cop = OnnxAdd('input', 'input')

    # Subgraph as a graph attribute.
    node = OnnxScan(cop, cop, output_names=['S1', 'S2'],
                    num_scan_inputs=1,
                    body=(scan_body.graph, [id_next, flat]))

    cop2 = OnnxIdentity(node[1], output_names=['cdist'])

    model_def = cop2.to_onnx(numpy.float32, numpy.float32)
    oinf = OnnxInference(model_def, inplace=False)

    print("DOT-SECTION", oinf.to_dot(recursive=True))
