"""
@file
@brief Command line about validation of prediction runtime.
"""
import os
from logging import getLogger
import warnings
from pandas import DataFrame
from sklearn.exceptions import ConvergenceWarning
from ..onnxrt.validate import enumerate_validated_operator_opsets, summary_report  # pylint: disable=E0402


def validate_runtime(verbose=1, opset_min=9, opset_max="",
                     check_runtime=True, runtime='python', debug=False,
                     models=None, out_raw="onnx_runtime_raw.xlsx",
                     out_summary="onnx_runtime_summary.xlsx",
                     dump_folder=None, benchmark=False,
                     catch_warnings=True, assume_finite=True,
                     versions=False, fLOG=print):
    """
    Walks through most of :epkg:`scikit-learn` operators
    or model or predictor or transformer, tries to convert
    them into :epkg:`ONNX` and computes the predictions
    with a specific runtime.

    :param verbose: integer from 0 (None) to 2 (full verbose)
    :param opset_min: tries every conversion from this minimum opset
    :param opset_max: tries every conversion up to maximum opset
    :param check_runtime: to check the runtime
        and not only the conversion
    :param runtime: runtime to check, python,
        onnxruntime1 to check :epkg:`onnxruntime`,
        onnxruntime2 to check every ONNX node independently
        with onnxruntime
    :param models: comma separated list of models to test or empty
        string to test them all
    :param debug: stops whenever an exception is raised
    :param out_raw: output raw results into this file (excel format)
    :param out_summary: output an aggregated view into this file (excel format)
    :param dump_folder: folder where to dump information (pickle)
        in case of mismatch
    :param benchmark: run benchmark
    :param catch_warnings: catch warnings
    :param assume_finite: See `config_context
        <https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html>`_,
        If True, validation for finiteness will be skipped, saving time, but leading
        to potential crashes. If False, validation for finiteness will be performed,
        avoiding error.
    :param versions: add columns with versions of used packages,
        :epkg:`numpy`, :epkg:`scikit-learn`, :epkg:`onnx`, :epkg:`onnxruntime`,
        :epkg:`sklearn-onnx`
    :param fLOG: logging function

    .. cmdref::
        :title: Valide a runtime against scikit-learn
        :cmd: -m mlprodict validate_runtime --help
        :lid: l-cmd-validate_runtime

        The command walks through all scikit-learn operators,
        tries to convert them, checks the predictions,
        and produces a report.

        Example::

            python -m mlprodict validate_runtime --models LogisticRegression,LinearRegression

        Following example benchmarks models
        :epkg:`sklearn:ensemble:RandomForestRegressor`,
        :epkg:`sklearn:tree:DecisionTreeRegressor`, it compares
        :epkg:`onnxruntime` against :epkg:`scikit-learn` for opset 10.

        ::

            python -m mlprodict validate_runtime -v 1 -o 10 -op 10 -c 1 -r onnxruntime1
                   -m RandomForestRegressor,DecisionTreeRegressor -out bench_onnxruntime.xlsx -b 1
    """
    models = None if models in (None, "") else models.strip().split(',')
    logger = getLogger('skl2onnx')
    logger.disabled = True
    if not dump_folder:
        dump_folder = None
    if dump_folder and not os.path.exists(dump_folder):
        raise FileNotFoundError("Cannot find dump_folder '{0}'.".format(
            dump_folder))
    if opset_max == "":
        opset_max = None
    if isinstance(opset_min, str):
        opset_min = int(opset_min)
    if isinstance(opset_max, str):
        opset_max = int(opset_max)
    if isinstance(verbose, str):
        verbose = int(verbose)

    def build_rows():
        rows = list(enumerate_validated_operator_opsets(
            verbose, models=models, fLOG=fLOG, runtime=runtime, debug=debug,
            dump_folder=dump_folder, opset_min=opset_min, opset_max=opset_max,
            benchmark=benchmark, assume_finite=assume_finite, versions=versions))
        return rows

    if catch_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",
                                  (UserWarning, ConvergenceWarning,
                                   RuntimeWarning, FutureWarning))
            rows = build_rows()
    else:
        rows = build_rows()

    df = DataFrame(rows)
    df.to_excel(out_raw, index=False)
    piv = summary_report(df)
    piv.to_excel(out_summary, index=False)
    if verbose > 0 and models is not None:
        fLOG(piv.T)
