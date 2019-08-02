"""
@file
@brief Command line about validation of prediction runtime.
"""
import os
from logging import getLogger
import warnings
from multiprocessing import Pool
from pandas import DataFrame
from sklearn.exceptions import ConvergenceWarning


def validate_runtime(verbose=1, opset_min=9, opset_max="",
                     check_runtime=True, runtime='python', debug=False,
                     models=None, out_raw="onnx_runtime_raw.xlsx",
                     out_summary="onnx_runtime_summary.xlsx",
                     dump_folder=None, benchmark=False,
                     catch_warnings=True, assume_finite=True,
                     versions=False, skip_models=None,
                     extended_list=True, separate_process=False,
                     fLOG=print):
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
    :param skip_models: models to skip
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
    :param extended_list: extends the list of :epkg:`scikit-learn` converters
        with converters implemented in this module
    :param separate_process: run every model in a separate process,
        this option must be used to run all model in one row
        even if one of them is crashing
    :param fLOG: logging function

    .. cmdref::
        :title: Validate a runtime against scikit-learn
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
    if separate_process:
        return _validate_runtime_separate_process(
            verbose=verbose, opset_min=opset_min, opset_max=opset_max,
            check_runtime=check_runtime, runtime=runtime, debug=debug,
            models=models, out_raw=out_raw,
            out_summary=out_summary,
            dump_folder=dump_folder, benchmark=benchmark,
            catch_warnings=catch_warnings, assume_finite=assume_finite,
            versions=versions, skip_models=skip_models,
            extended_list=extended_list,
            fLOG=fLOG)

    from ..onnxrt.validate import enumerate_validated_operator_opsets  # pylint: disable=E0402

    models = None if models in (None, "") else models.strip().split(',')
    skip_models = {} if skip_models in (
        None, "") else skip_models.strip().split(',')
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
    if isinstance(extended_list, str):
        extended_list = extended_list in ('1', 'True', 'true')

    def build_rows(models_):
        rows = list(enumerate_validated_operator_opsets(
            verbose, models=models_, fLOG=fLOG, runtime=runtime, debug=debug,
            dump_folder=dump_folder, opset_min=opset_min, opset_max=opset_max,
            benchmark=benchmark, assume_finite=assume_finite, versions=versions,
            extended_list=extended_list,
            filter_exp=lambda m, s: str(m) not in skip_models))
        return rows

    def catch_build_rows(models_):
        if catch_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",
                                      (UserWarning, ConvergenceWarning,
                                       RuntimeWarning, FutureWarning))
                rows = build_rows(models_)
        else:
            rows = build_rows(models_)
        return rows

    rows = catch_build_rows(models)
    return _finalize(rows, out_raw, out_summary, verbose, models, fLOG)


def _finalize(rows, out_raw, out_summary, verbose, models, fLOG):
    from ..onnxrt.validate import summary_report  # pylint: disable=E0402
    df = DataFrame(rows)
    if os.path.splitext(out_raw)[-1] == ".xlsx":
        df.to_excel(out_raw, index=False)
    else:
        df.to_csv(out_raw, index=False)
    piv = summary_report(df)
    if os.path.splitext(out_summary)[-1] == ".xlsx":
        piv.to_excel(out_summary, index=False)
    else:
        piv.to_csv(out_summary, index=False)
    if verbose > 0 and models is not None:
        fLOG(piv.T)

    # Drops data which cannot be serialized.
    for row in rows:
        keys = []
        for k in row:
            if 'lambda' in k:
                keys.append(k)
        for k in keys:
            del row[k]
    return rows


def _validate_runtime_dict(kwargs):
    return validate_runtime(**kwargs)


def _validate_runtime_separate_process(**kwargs):
    models = kwargs['models']
    if models in (None, ""):
        from ..onnxrt.validate_helper import sklearn_operators
        models = [_['name'] for _ in sklearn_operators(extended=True)]
    else:
        models = models.strip().split(',')

    skip_models = kwargs['skip_models']
    skip_models = {} if skip_models in (
        None, "") else skip_models.strip().split(',')

    verbose = kwargs['verbose']
    fLOG = kwargs['fLOG']
    all_rows = []
    skls = [m for m in models if m not in skip_models]
    skls.sort()

    if verbose > 0:
        from tqdm import tqdm
        pbar = tqdm(skls)
    else:
        pbar = skls

    for op in pbar:
        if not isinstance(pbar, list):
            pbar.set_description("[%s]" % (op + " " * (25 - len(op))))

        if kwargs['out_raw']:
            out_raw = os.path.splitext(kwargs['out_raw'])
            out_raw = "".join([out_raw[0], "_", op, out_raw[1]])
        else:
            out_raw = None

        if kwargs['out_summary']:
            out_summary = os.path.splitext(kwargs['out_summary'])
            out_summary = "".join([out_summary[0], "_", op, out_summary[1]])
        else:
            out_summary = None

        new_kwargs = kwargs.copy()
        if 'fLOG' in new_kwargs:
            del new_kwargs['fLOG']
        new_kwargs['out_raw'] = out_raw
        new_kwargs['out_summary'] = out_summary
        new_kwargs['models'] = op
        new_kwargs['verbose'] = 0  # tqdm fails

        p = Pool(1)
        try:
            result = p.apply_async(_validate_runtime_dict, [new_kwargs])
            lrows = result.get(timeout=150)  # timeout fixed to 150s
            all_rows.extend(lrows[0])
        except Exception as e:  # pylint: disable=W0703
            all_rows.append({
                'name': op, 'scenario': 'CRASH',
                'ERROR-msg': str(e).replace("\n", " -- ")
            })

    return _finalize(all_rows, kwargs['out_raw'], kwargs['out_summary'],
                     verbose, models, fLOG)
