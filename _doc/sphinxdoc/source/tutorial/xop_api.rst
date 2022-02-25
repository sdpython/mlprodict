
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
    from mlprodict.npy.xop import loadop

    # This line creates one class for the operator Sub and Mul.
    # It fails if the operators are misspelled.
    OnnxSub, OnnxMul = loadop('Sub', 'Mul')

    # Inputs are defined by their name as strings.
    diff = OnnxSub('X', 'Y')
    error = OnnxMul(diff, diff)

    # Then we create the ONNX graph defining 'X' and 'Y' as float.
    onx = error.to_onnx(numpy.float32, numpy.float32, verbose=5)

    # We check it does what it should
    X = numpy.array([4, 5], dtype=numpy.float32)
    Y = numpy.array([4.3, 5.7], dtype=numpy.float32)

    from onnxruntime import InferenceSession
    sess = InferenceSession(onx.SerializeToString())
    result = sess.run(None, {'X': X, 'Y': Y})
    assert_almost_equal((X - Y) ** 2, result[0])

    # Finally, we show the content of the graph.
    print(onnx_simple_text_plot(onx))

Visually, the model looks like the following.

.. gdot::
    :script: DOT-SECTION

    import numpy
    from numpy.testing import assert_almost_equal
    from mlprodict.plotting.text_plot import onnx_simple_text_plot
    from mlprodict.npy.xop import loadop

    # This line creates one class for the operator Sub and Mul.
    # It fails if the operators are misspelled.
    OnnxSub, OnnxMul = loadop('Sub', 'Mul')

    # Inputs are defined by their name as strings.
    diff = OnnxSub('X', 'Y')
    error = OnnxMul(diff, diff)

    # Then we create the ONNX graph defining 'X' and 'Y' as float.
    print(error)
    import pprint
    pprint.pprint(error.__dict__)
    onx = error.to_onnx(numpy.float32, numpy.float32, verbose=5)
    oinf = OnnxInference(onx, inplace=False)

    print("DOT-SECTION", oinf.to_dot())
