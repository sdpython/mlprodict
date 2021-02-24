# -*- coding: utf-8 -*-
"""
@file
@brief Main functions to convert machine learned model from *scikit-learn* model.
"""
from .g_sklearn_identify import identify_interpreter


def sklearn2graph(model, output_names=None, **kwargs):
    """
    Converts any kind of *scikit-learn* model into a *grammar* model.

    @param  model           scikit-learn model
    @param  output_names    names of the outputs
    @param  kwargs          additional parameters, sent to the converter
    @return                 converter to grammar model

    Short list of additional parameters:
    - *with_loop*: the pseudo code includes loops,
      this option is not available everywhere.

    If *output_names* is None, default values
    will be given to the inputs and outputs.
    One example on how to use this function.
    A *scikit-learn* model is trained and converted
    into a graph which implements the prediction
    function with the *grammar* language.

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

        # grammar is the expected scoring model.
        from mlprodict.grammar_sklearn import sklearn2graph
        gr = sklearn2graph(lr, output_names=['Prediction', 'Score'])

        # We can even check what the function should produce as a score.
        # Types are strict.
        import numpy
        X = numpy.array([[numpy.float32(1), numpy.float32(2)]])
        e2 = gr.execute(Features=X[0, :])
        print(e2)

        # We display the result in JSON.
        ser = gr.export(lang='json', hook={'array': lambda v: v.tolist(),
                                           'float32': lambda v: float(v)})
        import json
        print(json.dumps(ser, sort_keys=True, indent=2))

    For this particular example, the function is calling
    :func:`sklearn_logistic_regression <mlprodict.grammar_sklearn.sklearn_converters_linear_model.sklearn_logistic_regression>`
    and the code which produces the model looks like:

    ::

        model = LogisticRegression()
        model.fit(...)

        coef = model.coef_.ravel()
        bias = numpy.float32(model.intercept_[0])

        gr_coef = MLActionCst(coef)
        gr_var = MLActionVar(coef, input_names)
        gr_bias = MLActionCst(bias)
        gr_dot = MLActionTensorDot(gr_coef, gr_var)
        gr_dist = MLActionAdd(gr_dot, gr_bias)
        gr_sign = MLActionSign(gr_dist)
        gr_conc = MLActionConcat(gr_sign, gr_dist)
        gr_final = MLModel(gr_conc, output_names, name="LogisticRegression")

    The function interal represents any kind of function into a graph.
    This graph can easily exported in any format, :epkg:`Python` or any other programming
    language. The goal is not to evaluate it as it is slow due to the extra
    checkings ran all along the evaluation to make sure types are consistent.
    The current implementation supports conversion into C.

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

        # a grammar tree is the expected scoring model.
        from mlprodict.grammar_sklearn import sklearn2graph
        gr = sklearn2graph(lr, output_names=['Prediction', 'Score'])

        # We display the result in JSON.
        ccode = gr.export(lang='c')
        # We print after a little bit of cleaning.
        print("\\n".join(_ for _ in ccode['code'].split("\\n") if "//" not in _))

    Function ``adot``, ``sign``, ``concat`` are implemented in module
    :mod:`mlprodict.grammar_sklearn.cc.c_compilation`. Function
    :func:`compile_c_function <mlprodict.grammar_sklearn.cc.c_compilation.compile_c_function>`
    can compile this with :epkg:`cffi`.

    ::

        from mlprodict.grammar_sklearn.cc.c_compilation import compile_c_function
        fct = compile_c_function(code_c, 2)
        e2 = fct(X[0, :])
        print(e2)

    The output is the same as the prediction given by *scikit-learn*.
    """
    conv = identify_interpreter(model)
    return conv(model, output_names=output_names, **kwargs)
