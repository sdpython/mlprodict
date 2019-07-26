"""
@file
@brief Command line about validation of prediction runtime.
"""
import os
import pickle
from logging import getLogger
import warnings
import numpy
from pandas import read_csv
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx import to_onnx
from ..onnxrt import OnnxInference
from ..onnxrt.validate_difference import measure_relative_difference


def convert_validate(pkl, data, method="predict",
                     name='Y', outonnx="model.onnx",
                     runtime='python', metric="l1med",
                     use_double=None, noshape=False,
                     fLOG=print, verbose=1, register=True):
    """
    Converts a model stored in *pkl* file and measure the differences
    between the model and the ONNX predictions.

    :param pkl: pickle file
    :param data: data file, loaded with pandas,
        converted to a single array
    :param method: method to call
    :param name: output name
    :param outonnx: produced ONNX model
    :param runtime: runtime to use to compute predictions,
        'python', 'onnxruntime1' or 'onnxruntime2'
    :param metric: the metric 'l1med' is given by function
        :func:`measure_relative_difference
        <mlprodict.onnxrt.validate_difference.measure_relative_difference>`
    :param noshape: run the conversion with no shape information
    :param use_double: use double for the runtime if possible,
        two possible options, ``"float64"`` or ``'switch'``,
        the first option produces an ONNX file with doubles,
        the second option loads an ONNX file (float or double)
        and replaces matrices in ONNX with the matrices coming from
        the model, this second way is just for testing purposes
    :param verbose: verbose level
    :param register: registers additional converters implemented by this package
    :param fLOG: logging function
    :return: a dictionary with the results

    .. cmdref::
        :title: Convert and compare an ONNX file
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
        verbose = 0
    if use_double not in (None, 'float64', 'switch'):
        raise ValueError(
            "use_double must be either None, 'float64' or 'switch'")
    if verbose == 0:
        logger = getLogger('skl2onnx')
        logger.disabled = True
    if not os.path.exists(pkl):
        raise FileNotFoundError("Unable to find model '{}'.".format(pkl))
    if not os.path.exists(data):
        raise FileNotFoundError("Unable to find data '{}'.".format(data))
    if os.path.exists(outonnx):
        warnings.warn("File '{}' will be overwritten.".format(outonnx))
    if verbose > 0:
        fLOG("[convert_validate] load model '{}'".format(pkl))
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    if verbose > 0:
        fLOG("[convert_validate] load data '{}'".format(data))
    df = read_csv(data)
    if verbose > 0:
        fLOG("[convert_validate] convert data into matrix")

    if use_double == 'float64':
        dtype = numpy.float64
        tensor_type = DoubleTensorType
    else:
        dtype = numpy.float32
        tensor_type = FloatTensorType

    if register:
        from ..onnx_conv import register_converters
        register_converters()

    numerical = df.values.astype(dtype)
    if noshape:
        if verbose > 0:
            fLOG("[convert_validate] convert the model with no shape information")
        onx = to_onnx(model, initial_types=[
                      ('X', tensor_type([None, None]))],
                      dtype=dtype)
    else:
        if verbose > 0:
            fLOG("[convert_validate] convert the model with shapes")
        onx = to_onnx(model, numerical, dtype=dtype)
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
        raise ValueError("Number of methods and outputs do not match: {}, {}".format(
            names, methods))

    if metric != 'l1med':
        raise ValueError("Unknown metric '{}'".format(metric))

    if verbose > 0:
        fLOG("[convert_validate] compute predictions from ONNX with name '{}'".format(
            name))
    ort_preds = sess.run({'X': numerical})

    metrics = []
    out_skl_preds = []
    out_ort_preds = []
    for method_, name_ in zip(methods, names):
        if verbose > 0:
            fLOG("[convert_validate] compute predictions with method '{}'".format(
                method_))
        meth = getattr(model, method_)
        skl_pred = meth(numerical)
        out_skl_preds.append(skl_pred)

        if name_ not in ort_preds:
            raise KeyError("Unable to find output name '{}' in {}".format(
                name_, list(sorted(ort_preds))))

        ort_pred = ort_preds[name_]
        out_ort_preds.append(ort_pred)
        diff = measure_relative_difference(skl_pred, ort_pred)
        if verbose > 0:
            fLOG("[convert_validate] {}={}".format(metric, diff))
        metrics.append(diff)

    return dict(skl_pred=out_skl_preds, ort_pred=out_ort_preds,
                metrics=metrics, onnx=memory)
