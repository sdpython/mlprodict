"""
@file
@brief Command line about validation of prediction runtime.
"""
import os
from logging import getLogger
from pandas import DataFrame
from ..onnxrt.validate import enumerate_validated_operator_opsets, summary_report  # pylint: disable=E0402


def validate_runtime(verbose=1, opset_min=9, opset_max=11,
                     check_runtime=True, runtime='CPU', debug=False,
                     models=None, out_raw="onnx_runtime_raw.xlsx",
                     out_summary="onnx_runtime_summart.xlsx",
                     dump_folder=None, fLOG=print):
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
    :param runtime: runtime to check, CPU for python,
        onnxruntime to check every ONNX node independenlty
        or onnxruntime-whole to check :epkg:`onnxruntime`
    :param models: comma separated list of models to test or empty
        string to test them all
    :param debug: stops whenever an exception is raised
    :param out_raw: output raw results into this file (excel format)
    :param out_summary: output an aggregated view into this file (excel format)
    :param dump_folder: folder where to dump information (pickle)
        in case of mismatch
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
    """
    models = None if models in (None, "") else models.strip().split(',')
    logger = getLogger('skl2onnx')
    logger.disabled = True
    if not dump_folder:
        dump_folder = None
    if dump_folder and not os.path.exists(dump_folder):
        raise FileNotFoundError(dump_folder)
    rows = list(enumerate_validated_operator_opsets(verbose, models=models, fLOG=fLOG,
                                                    runtime=runtime, debug=debug,
                                                    dump_folder=dump_folder))
    df = DataFrame(rows)
    df.to_excel(out_raw, index=False)
    piv = summary_report(df)
    piv.to_excel(out_summary, index=False)
    if verbose > 0 and models is not None:
        fLOG(piv.T)
