"""
@file
@brief Command line about validation of prediction runtime.
"""
from logging import getLogger
from pandas import DataFrame
from ..onnxrt.validate import validate_operator_opsets, summary_report  # pylint: disable=E0402


def validate_runtime(verbose=1, opset_min=9, opset_max=11,
                     check_runtime=True, runtime='CPU',
                     out_raw="onnx_runtime_raw.xlsx",
                     out_summary="onnx_runtime_summart.xlsx"):
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
        or onnxruntime-whole to check :epkg:`onnxruntime`,
    :param out_raw: output raw results into this file (excel format)
    :param out_summary: output an aggregated view into this file (excel format)
    """
    logger = getLogger('skl2onnx')
    logger.disabled = True
    verbose = 1 if __name__ == "__main__" else 0
    rows = validate_operator_opsets(verbose, debug=None, fLOG=print,
                                    runtime=runtime)
    df = DataFrame(rows)
    df.to_excel(out_raw, index=False)
    piv = summary_report(df)
    piv.to_excel(out_summary, index=False)
