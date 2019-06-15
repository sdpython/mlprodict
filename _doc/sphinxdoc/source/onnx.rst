
.. _l-onnx-pyrun:

ONNX
====

This module implements a python runtime for :epkg:`ONNX`.
It is a work constantly in progress. It was started to
facilitate the implementation of :epkg:`scikit-learn`
converters in :epkg:`sklearn-onnx`.
Main class is :class:`OnnxInference
<mlprodict.onnxrt.onnx_inference.OnnxInference>`.

.. runpython::
    :showcode:

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, _ = train_test_split(X, y)
    clr = LinearRegression()
    clr.fit(X_train, y_train)

    exp = clr.predict(X_test[:5])
    print(exp)

    model_def = to_onnx(clr, X_train.astype(numpy.float32))
    oinf = OnnxInference(model_def)
    y = oinf.run({'X': X_test[:5]})
    print(y)

Some ONNX operators converters are using were not all
available in older version of :epkg:`ONNX`. This version is called
*opset number*. :epkg:`ONNX` 1.4.0 is opset 9,
:epkg:`ONNX` 1.5.0 is opset 10...
Next table shows which operator is available in which opset.
An empty cell means it is not available. Other cells
contains concatenated flags whose meaning is the following:

* ``ERROR`` means the automated process failed to give
  a appropriate status or the runtime produces predictions
  too far from the original predictions,
  the second part of the constant gives an
  approximate diagnostic, last columns gives the exception
  message,
* ``OK``: the converter works fine and the runtime produces
  predictions almost equal to the orignal predictions,
  absolute difference is below :math:`1e-5`,
* ``e<%f``: the converter works fine and the runtime produces
  predictions close to the orignal predictions,
  absolute difference is below the threshold,
* ``i|j``: the model was converted for a specific opset but
  the converted ONNX is compatible with smaller opset,
  *i* is the smallest compatible opset for the main domain,
  *j* is the smallest compatible opset for the ai domain,
* ``NOBATCH``: the runtime is unable to compute the predictions
  for multiple observations at the same time, it needs to be
  called for each observation.

The model are tested through simple problems using the Iris dataset.
The datasets is split into train test datasets.

* *bin-class*: binary classification,
* *multi-class*: multi-class classification,
* *regression*: regression,
* *num-transform*: no label, only numerical features

The following table tracks what is available,
what is working.

.. runpython::
    :showcode:

    from logging import getLogger
    from pyquickhelper.loghelper import noLOG
    from pandas import DataFrame
    from pyquickhelper.texthelper import df2rst
    from mlprodict.onnxrt.validate import validate_operator_opsets, summary_report

    logger = getLogger('skl2onnx')
    logger.disabled = True
    rows = validate_operator_opsets(0, debug=None, fLOG=noLOG)
    df = DataFrame(rows)
    piv = summary_report(df)
    print(df2rst(piv))
