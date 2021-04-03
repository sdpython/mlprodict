"""
@file
@brief Complex but recurring testing functions.
"""
import random
import pandas
import numpy
from numpy.testing import assert_allclose
from ..grammar_sklearn import sklearn2graph
from ..grammar_sklearn.cc import compile_c_function


def iris_data():
    """
    Returns ``(X, y)`` for iris data.
    """
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data[:, :2]
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    y = iris.target
    return X, y


def check_is_almost_equal(xv, exp, precision=1e-5, message=None):
    """
    Checks that two floats or two arrays are almost equal.

    @param  xv          float or vector
    @param  exp         expected value
    @param  precision   precision
    @param  message     additional message
    """
    if isinstance(exp, float) or len(exp.ravel()) == 1:
        if not (isinstance(xv, float) or len(xv.ravel()) == 1):
            raise TypeError(  # pragma: no cover
                "Type mismatch between {0} and {1} (expected).".format(
                    type(xv), type(exp)))
        diff = abs(xv - exp)
        if diff > 1e-5:
            raise ValueError(  # pragma: no cover
                "Predictions are different expected={0}, computed={1}".format(
                    exp, xv))
    else:
        if not isinstance(xv, numpy.ndarray):
            raise TypeError(
                "Type mismatch between {0} and {1} (expected).".format(type(xv), type(exp)))
        xv = xv.ravel()
        exp = exp.ravel()
        try:
            assert_allclose(xv, exp, atol=precision)
        except AssertionError as e:
            if message is None:
                raise e
            else:
                raise AssertionError(message) from e  # pragma: no cover


def check_model_representation(model, X, y=None, convs=None,
                               output_names=None, only_float=True,
                               verbose=False, suffix="", fLOG=None):
    """
    Checks that a trained model can be exported in a specific list
    of formats and produces the same outputs if the
    representation can be used to predict.

    @param  model           model (a class or an instance of a model but not trained)
    @param  X               features
    @param  y               targets
    @param  convs           list of format to check, all possible by default ``['json', 'c']``
    @param  output_names    list of output columns
                            (can be None, a default value is infered based on scikit-learn output then)
    @param  verbose         print some information
    @param  suffix          add this to disambiguate module
    @param  fLOG            logging function
    @return                 function to call to run the prediction
    """
    if not only_float:
        raise NotImplementedError(  # pragma: no cover
            "Only float are allowed.")
    if isinstance(X, list):
        X = pandas.DataFrame(X)
        if len(X.shape) != 2:
            raise ValueError(  # pragma: no cover
                "X cannot be converted into a proper DataFrame. It has shape {0}."
                "".format(X.shape))
        if only_float:
            X = X.values
    if isinstance(y, list):
        y = numpy.array(y)
    if convs is None:
        convs = ['json', 'c']

    # sklearn
    if not hasattr(model.__class__, "fit"):
        # It is a class object and not an instance.
        # We use the default values.
        model = model()

    model.fit(X, y)
    h = random.randint(0, X.shape[0] - 1)
    if isinstance(X, pandas.DataFrame):
        oneX = X.iloc[h, :].astype(numpy.float32)
    else:
        oneX = X[h, :].ravel().astype(numpy.float32)

    # model or transform
    moneX = numpy.resize(oneX, (1, len(oneX)))
    if hasattr(model, "predict"):
        ske = model.predict(moneX)
    else:
        ske = model.transform(moneX)

    if verbose and fLOG:
        fLOG("---------------------")
        fLOG(type(oneX), oneX.dtype)
        fLOG(model)
        for k, v in sorted(model.__dict__.items()):
            if k[-1] == '_':
                fLOG("  {0}={1}".format(k, v))
        fLOG("---------------------")

    # grammar
    gr = sklearn2graph(model, output_names=output_names)
    lot = gr.execute(Features=oneX)
    if verbose and fLOG:
        fLOG(gr.graph_execution())

    # verification
    check_is_almost_equal(lot, ske)

    # default for output_names
    if output_names is None:
        if len(ske.shape) == 1:
            output_names = ["Prediction"]
        elif len(ske.shape) == 2:
            output_names = ["p%d" % i for i in range(ske.shape[1])]
        else:
            raise ValueError(  # pragma: no cover
                "Cannot guess default values for output_names.")

    for lang in convs:
        if lang in ('c', ):
            code_c = gr.export(lang=lang)['code']
            if code_c is None:
                raise ValueError("cannot be None")  # pragma: no cover

            compile_fct = compile_c_function

            from contextlib import redirect_stdout, redirect_stderr
            from io import StringIO
            fout = StringIO()
            ferr = StringIO()
            with redirect_stdout(fout):
                with redirect_stderr(ferr):
                    try:
                        fct = compile_fct(
                            code_c, len(output_names), suffix=suffix, fLOG=lambda s: fout.write(s + "\n"))
                    except Exception as e:  # pragma: no cover
                        raise RuntimeError(
                            "Unable to compile a code\n-OUT-\n{0}\n-ERR-\n{1}\n-CODE-"
                            "\n{2}".format(fout.getvalue(), ferr.getvalue(), code_c)) from e

            if verbose and fLOG:
                fLOG("-----------------")
                fLOG(output_names)
                fLOG("-----------------")
                fLOG(code_c)
                fLOG("-----------------")
                fLOG("h=", h, "oneX=", oneX)
                fLOG("-----------------")
            lotc = fct(oneX)
            check_is_almost_equal(
                lotc, ske, message="Issue with lang='{0}'".format(lang))
            lotc_exp = lotc.copy()
            lotc2 = fct(oneX, lotc)
            if not numpy.array_equal(lotc_exp, lotc2):
                raise ValueError(  # pragma: no cover
                    "Second call returns different results.\n{0}\n{1}".format(
                        lotc_exp, lotc2))
        else:
            ser = gr.export(lang="json", hook={'array': lambda v: v.tolist()})
            if ser is None:
                raise ValueError(  # pragma: no cover
                    "No output for long='{0}'".format(lang))
