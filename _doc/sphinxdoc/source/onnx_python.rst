
.. _l-onnx-pyrun-tbl:

Python Runtime
==============

This module implements a python runtime for :epkg:`ONNX`.
It is a work constantly in progress. It was started to
facilitate the implementation of :epkg:`scikit-learn`
converters in :epkg:`sklearn-onnx`.
Main class is :class:`OnnxInference
<mlprodict.onnxrt.onnx_inference.OnnxInference>`.

.. runpython::
    :showcode:

    import numpy
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from skl2onnx import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, _ = train_test_split(X, y)
    clr = LinearRegression()
    clr.fit(X_train, y_train)

    # predictions with scikit-learn
    exp = clr.predict(X_test[:5])
    print(exp)

    # predictions with onnxruntime
    model_def = to_onnx(clr, X_train.astype(numpy.float32))
    oinf = OnnxInference(model_def)
    y = oinf.run({'X': X_test[:5]})
    print(y)

Some :epkg:`ONNX` operators converters are using were not all
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
  relative difference is below :math:`1e-5`,
* ``e<%f``: the converter works fine and the runtime produces
  predictions close to the orignal predictions,
  relative difference is below the threshold,
* ``i|j``: the model was converted for a specific opset but
  the converted ONNX is compatible with smaller opset,
  *i* is the smallest compatible opset for the main domain,
  *j* is the smallest compatible opset for the ai domain,
* ``NOBATCH``: the runtime is unable to compute the predictions
  for multiple observations at the same time, it needs to be
  called for each observation.

The model are tested through simple problems using the Iris dataset.
The datasets is split into train test datasets.
Function :func:`find_suitable_problem
<mlprodict.onnxrt.validate_problems.find_suitable_problem>` gives
the list of problem every :epkg:`scikit-learn` is tested on.
The main ones are the following:

* *bin-class*: binary classification,
* *multi-class*: multi-class classification,
* *regression*: regression,
* *cluster*: clutering,
* *outlier*: outlier detection,
* *num-transform*: no label, only numerical features

Next table tracks what is available,
what is working and some indication about
the cause of the error if it does not work.

.. runpython::
    :showcode:
    :rst:
    :warningout: PendingDeprecationWarning UserWarning RuntimeWarning

    from logging import getLogger
    from pyquickhelper.loghelper import noLOG
    from pandas import DataFrame
    from pyquickhelper.pandashelper import df2rst
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.utils.testing import ignore_warnings
    from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning, FutureWarning))
    def build_table():
        logger = getLogger('skl2onnx')
        logger.disabled = True
        rows = list(enumerate_validated_operator_opsets(0, debug=None, fLOG=noLOG,
                                                        models=['LinearRegression',
                                                                'LogisticRegression'],
                                                        benchmark=True))
        df = DataFrame(rows)
        piv = summary_report(df)

        if "ERROR-msg" in piv.columns:
            def shorten(text):
                text = str(text)
                if len(text) > 75:
                    text = text[:75] + "..."
                return text

            piv["ERROR-msg"] = piv["ERROR-msg"].apply(shorten)

        print(df2rst(piv, number_format=2,
                     replacements={'nan': '', 'ERR: 4convert': ''}))

    build_table()

Full results are available at :ref:`l-onnx-bench-python`.
