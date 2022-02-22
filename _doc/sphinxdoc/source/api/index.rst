
API
===

This is a summary of functions this modules provides.

**ONNX converters**

.. toctree::
    :maxdepth: 1

    onnx_conv
    sklapi

**Write ONNX graphs**

.. toctree::
    :maxdepth: 1

    npy
    xop
    ast

**ONNX runtime**

.. toctree::
    :maxdepth: 1

    onnxrt
    onnxrt_ops
    testing

**ONNX validation, benchmark, tools**

.. toctree::
    :maxdepth: 1

    asv
    validation
    tools

**Outside ONNX world**

This was a first experiment to play with machine learning:
convert a model into :epkg:`C` code. A similar way than
:epkg:`ONNX` but far less advanced.

.. toctree::
    :maxdepth: 1

    cc_grammar

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    y[y == 2] = 1
    lr = LogisticRegression()
    lr.fit(X, y)

    # Conversion into a graph.
    from mlprodict.grammar_sklearn import sklearn2graph
    gr = sklearn2graph(lr, output_names=['Prediction', 'Score'])

    # Conversion into C
    ccode = gr.export(lang='c')
    # We print after a little bit of cleaning (remove all comments)
    print("\n".join(_ for _ in ccode['code'].split("\n") if "//" not in _))
