
onnxruntime2: independent onnxruntime for every node
====================================================

This runtime does not load the :epkg:`ONNX` data in a single
session but instead calls :epkg:`onnxruntime` for each node
independently. This was developped mostly to facilitate
the implementation of converters from :epkg:`scikit-learn`
object to :epkg:`ONNX`. We create a table similar to
:ref:`l-onnx-pyrun-tbl`.

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
                                                        runtime='onnxruntime2',
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

Full results are available at :ref:`l-onnx-bench-onnxruntime`.
