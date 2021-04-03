"""
@file
@brief Command line about validation of prediction runtime.
"""
import os
import pickle
from logging import getLogger
import warnings
from pandas import read_csv
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from ..onnx_conv import to_onnx
from ..onnxrt import OnnxInference
from ..onnxrt.optim import onnx_optimisations
from ..onnxrt.validate.validate_difference import measure_relative_difference
from ..onnx_conv import guess_schema_from_data, guess_schema_from_model


def convert_validate(pkl, data=None, schema=None,
                     method="predict", name='Y',
                     target_opset=None,
                     outonnx="model.onnx",
                     runtime='python', metric="l1med",
                     use_double=None, noshape=False,
                     optim='onnx', rewrite_ops=True,
                     options=None, fLOG=print, verbose=1,
                     register=True):
    """
    Converts a model stored in *pkl* file and measure the differences
    between the model and the ONNX predictions.

    :param pkl: pickle file
    :param data: data file, loaded with pandas,
        converted to a single array, the data is used to guess
        the schema if *schema* not specified
    :param schema: initial type of the model
    :param method: method to call
    :param name: output name
    :param target_opset: target opset
    :param outonnx: produced ONNX model
    :param runtime: runtime to use to compute predictions,
        'python', 'python_compiled',
        'onnxruntime1' or 'onnxruntime2'
    :param metric: the metric 'l1med' is given by function
        :func:`measure_relative_difference
        <mlprodict.onnxrt.validate.validate_difference.measure_relative_difference>`
    :param noshape: run the conversion with no shape information
    :param use_double: use double for the runtime if possible,
        two possible options, ``"float64"`` or ``'switch'``,
        the first option produces an ONNX file with doubles,
        the second option loads an ONNX file (float or double)
        and replaces matrices in ONNX with the matrices coming from
        the model, this second way is just for testing purposes
    :param optim: applies optimisations on the first ONNX graph,
        use 'onnx' to reduce the number of node Identity and
        redundant subgraphs
    :param rewrite_ops: rewrites some converters from skl2onnx
    :param options: additional options for conversion,
        dictionary as a string
    :param verbose: verbose level
    :param register: registers additional converters implemented by this package
    :param fLOG: logging function
    :return: a dictionary with the results

    .. cmdref::
        :title: Converts and compares an ONNX file
        :cmd: -m mlprodict convert_validate --help
        :lid: l-cmd-convert_validate

        The command converts and validates a :epkg:`scikit-learn` model.
        An example to check the prediction of a logistic regression.

        ::

            import os
            import pickle
            import pandas
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from mlprodict.__main__ import main
            from mlprodict.cli import convert_validate

            iris = load_iris()
            X, y = iris.data, iris.target
            X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
            clr = LogisticRegression()
            clr.fit(X_train, y_train)

            pandas.DataFrame(X_test).to_csv("data.csv", index=False)
            with open("model.pkl", "wb") as f:
                pickle.dump(clr, f)

        And the command line to check the predictions
        using a command line.

        ::

            convert_validate --pkl model.pkl --data data.csv
                             --method predict,predict_proba
                             --name output_label,output_probability
                             --verbose 1
    """
    if fLOG is None:
        verbose = 0  # pragma: no cover
    if use_double not in (None, 'float64', 'switch'):
        raise ValueError(  # pragma: no cover
            "use_double must be either None, 'float64' or 'switch'")
    if optim == '':
        optim = None  # pragma: no cover
    if target_opset == '':
        target_opset = None  # pragma: no cover
    if verbose == 0:
        logger = getLogger('skl2onnx')
        logger.disabled = True
    if not os.path.exists(pkl):
        raise FileNotFoundError(  # pragma: no cover
            "Unable to find model '{}'.".format(pkl))
    if os.path.exists(outonnx):
        warnings.warn("File '{}' will be overwritten.".format(outonnx))
    if verbose > 0:
        fLOG("[convert_validate] load model '{}'".format(pkl))
    with open(pkl, "rb") as f:
        model = pickle.load(f)

    if use_double == 'float64':
        tensor_type = DoubleTensorType
    else:
        tensor_type = FloatTensorType
    if options in (None, ''):
        options = None
    else:
        from ..onnxrt.validate.validate_scenarios import (
            interpret_options_from_string)
        options = interpret_options_from_string(options)
        if verbose > 0:
            fLOG("[convert_validate] options={}".format(repr(options)))

    if register:
        from ..onnx_conv import (
            register_converters, register_rewritten_operators)
        register_converters()
        register_rewritten_operators()

    # data and schema
    if data is None or not os.path.exists(data):
        if schema is None:
            schema = guess_schema_from_model(model, tensor_type)
        if verbose > 0:
            fLOG("[convert_validate] model schema={}".format(schema))
        df = None
    else:
        if verbose > 0:
            fLOG("[convert_validate] load data '{}'".format(data))
        df = read_csv(data)
        if verbose > 0:
            fLOG("[convert_validate] convert data into matrix")
        if schema is None:
            schema = guess_schema_from_data(df, tensor_type)
        if schema is None:
            schema = [  # pragma: no cover
                ('X', tensor_type([None, df.shape[1]]))]
        if len(schema) == 1:
            df = df.values
        if verbose > 0:
            fLOG("[convert_validate] data schema={}".format(schema))

    if noshape:
        if verbose > 0:
            fLOG(  # pragma: no cover
                "[convert_validate] convert the model with no shape information")
        schema = [(name, col.__class__([None, None])) for name, col in schema]
        onx = to_onnx(
            model, initial_types=schema, rewrite_ops=rewrite_ops,
            target_opset=target_opset, options=options)
    else:
        if verbose > 0:
            fLOG("[convert_validate] convert the model with shapes")
        onx = to_onnx(
            model, initial_types=schema, target_opset=target_opset,
            rewrite_ops=rewrite_ops, options=options)

    if optim is not None:
        if verbose > 0:
            fLOG("[convert_validate] run optimisations '{}'".format(optim))
        onx = onnx_optimisations(onx, optim=optim)
    if verbose > 0:
        fLOG("[convert_validate] saves to '{}'".format(outonnx))
    memory = onx.SerializeToString()
    with open(outonnx, 'wb') as f:
        f.write(memory)

    if verbose > 0:
        fLOG("[convert_validate] creates OnnxInference session")
    sess = OnnxInference(onx, runtime=runtime)
    if use_double == "switch":
        if verbose > 0:
            fLOG("[convert_validate] switch to double")
        sess.switch_initializers_dtype(model)

    if verbose > 0:
        fLOG("[convert_validate] compute prediction from model")

    if ',' in method:
        methods = method.split(',')
    else:
        methods = [method]
    if ',' in name:
        names = name.split(',')
    else:
        names = [name]

    if len(names) != len(methods):
        raise ValueError(
            "Number of methods and outputs do not match: {}, {}".format(
                names, methods))

    if metric != 'l1med':
        raise ValueError(  # pragma: no cover
            "Unknown metric '{}'".format(metric))

    if df is None:
        # no test on data
        return dict(onnx=memory)

    if verbose > 0:
        fLOG("[convert_validate] compute predictions from ONNX with name '{}'"
             "".format(name))

    ort_preds = sess.run(
        {'X': df}, verbose=max(verbose - 1, 0), fLOG=fLOG)

    metrics = []
    out_skl_preds = []
    out_ort_preds = []
    for method_, name_ in zip(methods, names):
        if verbose > 0:
            fLOG("[convert_validate] compute predictions with method '{}'".format(
                method_))
        meth = getattr(model, method_)
        skl_pred = meth(df)
        out_skl_preds.append(df)

        if name_ not in ort_preds:
            raise KeyError(
                "Unable to find output name '{}' in {}".format(
                    name_, list(sorted(ort_preds))))

        ort_pred = ort_preds[name_]
        out_ort_preds.append(ort_pred)
        diff = measure_relative_difference(skl_pred, ort_pred)
        if verbose > 0:
            fLOG("[convert_validate] {}={}".format(metric, diff))
        metrics.append(diff)

    return dict(skl_pred=out_skl_preds, ort_pred=out_ort_preds,
                metrics=metrics, onnx=memory)
