"""
Extensions for mlprodict.
"""
import os
from textwrap import dedent
from logging import getLogger
from pandas import DataFrame, read_excel, read_csv, concat, Series
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
import sphinx
from tqdm import tqdm
from pyquickhelper.loghelper import noLOG
from pyquickhelper.pycode import is_travis_or_appveyor
from pyquickhelper.pandashelper import df2rst
from pyquickhelper.loghelper import run_cmd
from pyquickhelper.loghelper.run_cmd import get_interpreter_path
from mlprodict.onnxrt.validate.validate_helper import sklearn_operators
from mlprodict.onnxrt.doc.doc_write_helper import (
    split_columns_subsets, build_key_split, filter_rows, _make_opset)
from mlprodict.onnxrt.validate.validate_summary import _clean_values_optim
from mlprodict.onnx_conv import register_converters, register_rewritten_operators
register_converters()
try:
    register_rewritten_operators()
except KeyError:
    import warnings
    warnings.warn("converter for HistGradientBoosting* not not exist. "
                  "Upgrade sklearn-onnx")


@ignore_warnings(category=(UserWarning, ConvergenceWarning,
                           RuntimeWarning, FutureWarning))
def write_page_onnxrt_ops(app):
    from mlprodict.onnxrt.doc.doc_write_helper import compose_page_onnxrt_ops
    logger = getLogger('mlprodict')
    srcdir = app.builder.srcdir if app is not None else ".."
    whe = os.path.join(os.path.abspath(srcdir), "api", "onnxrt_ops.rst")
    logger.info("[mlprodict] create page '{}'.".format(whe))
    print("[mlprodict-sphinx] create page '{}'.".format(whe))
    page = compose_page_onnxrt_ops()
    with open(whe, "w", encoding='utf-8') as f:
        f.write(page)
    print("[mlprodict-sphinx] done page '{}'.".format(whe))


def write_page_onnxrt_benches(app, runtime, skip=None, white_list=None):

    from mlprodict.onnxrt.validate.validate import enumerate_validated_operator_opsets
    logger = getLogger('mlprodict')
    srcdir = app.builder.srcdir if app is not None else ".."

    if runtime in ('python', 'python_compiled'):
        whe = os.path.join(os.path.abspath(srcdir),
                           "skl_converters", "bench_python.rst")
    elif runtime == 'onnxruntime2':
        whe = os.path.join(os.path.abspath(srcdir),
                           "skl_converters", "bench_onnxrt2.rst")
    elif runtime == 'onnxruntime1':
        whe = os.path.join(os.path.abspath(srcdir),
                           "skl_converters", "bench_onnxrt1.rst")
    else:
        raise RuntimeError("Unsupported runtime '{}'.".format(runtime))

    logger.info("[mlprodict] create page '{}'.".format(whe))
    print("[mlprodict-sphinx] create page runtime '{}' - '{}'.".format(runtime, whe))

    out_sum = os.path.join(
        srcdir, "skl_converters", "bench_raw_%s.xlsx" % runtime)
    if not os.path.exists(out_sum):
        raise FileNotFoundError("Unable to find %r." % out_sum)
    piv = pandas.from_excel(out_sum, index=False)
    logger.info("[mlprodict] read '{}'.".format(out_sum))
    print("[mlprodict-sphinx] read '{}'".format(out_sum))

    out_raw = os.path.join(
        srcdir, "skl_converters", "bench_raw_%s.xlsx" % runtime)
    if not os.path.exists(out_raw):
        raise FileNotFoundError("Unable to find %r." % out_raw)
    df_raw = pandas.to_excel(out_raw, index=False)
    logger.info("[mlprodict] wrote '{}'.".format(out_raw))
    print("[mlprodict-sphinx] wrote '{}'".format(out_raw))

    logger.info("[mlprodict] shape '{}'.".format(piv.shape))
    print("[mlprodict-sphinx] shape '{}'".format(piv.shape))

    def make_link(row):
        link = ":ref:`{name} <l-{name}-{problem}-{scenario}-{optim}-{opset}>`"
        name = row['name']
        problem = row['problem']
        scenario = row['scenario']
        optim = _clean_values_optim(
            str(row.get('optim', '')).replace("nan", ""))
        optim = optim.replace(" ", "").replace(
            "{", "").replace("}", "").replace("'", "")
        opset = _make_opset(row)
        return link.format(name=name, problem=problem,
                           scenario=scenario, optim=optim,
                           opset=opset)

    piv['name'] = piv.apply(lambda row: make_link(row), axis=1)
    piv.reset_index(drop=True, inplace=True)

    if "ERROR-msg" in piv.columns:
        def shorten(text):
            text = str(text)
            if len(text) > 75:
                text = text[:75] + "..."
            return text

        piv["ERROR-msg"] = piv["ERROR-msg"].apply(shorten)

    logger.info("[mlprodict] write '{}'.".format(whe))
    print("[mlprodict-sphinx] write '{}'".format(whe))

    with open(whe, 'w', encoding='utf-8') as f:
        title = "Availability of scikit-learn model for runtime {0}".format(
            runtime)
        f.write(dedent('''
        .. _l-onnx-bench-{0}:

        {1}
        {2}

        The following metrics measure the ratio between the prediction time
        for the runtime compare to :epkg:`scikit-learn`.
        It gives an order of magnitude. They are done by setting
        ``assume_finite=True`` (see `config_context
        <https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html>`_).
        The computed ratio is:

        .. math::

            \\frac{{\\textit{{execution when predicting with a custom ONNX runtime}}}}
            {{\\textit{{execution when predicting with scikit-learn (assume\\_finite=True)}}}}

        Due to float32 conversion, it may happen than the highest difference
        is quite high. The proposition :math:`a < b \\Rightarrow [a] < [b]`
        is usually true and but not true all the time. It is the same after number
        where rounded to float32, that's why the result considers the
        fourth highest difference and not the first three.

        Some figures are missing when the number of observations is high.
        That means the prediction is slow for one of the runtime
        (ONNX, scikit-learn) and it would take too long to go further.
        The list of problems can be found in the documentation of
        function :func:`find_suitable_problem
        <mlprodict.onnxrt.validate.validate_problems.find_suitable_problem>`.
        Default values are usually used to create models but other
        scenarios are defined by :func:`build_custom_scenarios
        <mlprodict.onnxrt.validate.validate_scenarios.build_custom_scenarios>`
        and :func:`build_custom_scenarios (2)
        <from mlprodict.onnxrt.validate.validate_scenarios.build_custom_scenarios>`.
        The benchmark can be generated with a command line:

        ::

            python -m mlprodict validate_runtime --verbose=1 --out_raw=data.csv --out_summary=summary.xlsx --benchmark=1 --dump_folder=. --runtime={0}

        The option ``-se 1`` may be used if the process crashes. The command line
        can also be extended to test only one model or to skip another one. The whole
        batch takes between 5 and 15 minutes depending on the machine.

        Full data: :download:`{3} <../{3}>`

        .. contents::
            :local:

        '''.format(runtime, title, "=" * len(title),
                   "bench_sum_%s.xlsx" % runtime)))
        common, subsets = split_columns_subsets(piv)
        f.write(df2rst(piv, number_format=2,
                       replacements={'nan': '', 'ERR: 4convert': ''},
                       split_row=lambda index, dp=piv: build_key_split(
                           dp.loc[index, "name"], index),
                       split_col_common=common,
                       split_col_subsets=subsets,
                       filter_rows=filter_rows,
                       column_size={'problem': 25},
                       label_pattern=".. _lpy-{section}:"))
    logger.info(
        "[mlprodict] done page '{}'.".format(whe))
    print("[mlprodict-sphinx] done page runtime '{}' - '{}'.".format(runtime, whe))


def write_page_onnxrt_benches_python(app, white_list=None):
    write_page_onnxrt_benches(app, 'python_compiled', white_list=white_list)


def write_page_onnxrt_benches_onnxruntime1(app, white_list=None):
    write_page_onnxrt_benches(app, 'onnxruntime1', white_list=white_list)


def write_page_onnxrt_benches_onnxruntime2(app, white_list=None):
    write_page_onnxrt_benches(app, 'onnxruntime2', white_list=white_list)


def setup(app):
    """
    Preparation of the documentation.
    """
    app.connect('builder-inited', write_page_onnxrt_benches_python)
    app.connect('builder-inited', write_page_onnxrt_benches_onnxruntime1)
    # app.connect('builder-inited', write_page_onnxrt_benches_onnxruntime2)
    app.connect('builder-inited', write_page_onnxrt_ops)
    return {'version': sphinx.__display_version__,
            'parallel_read_safe': False,
            'parallel_write_safe': False}


if __name__ == '__main__':
    # write_page_onnxrt_benches_python(None, white_list={'AdaBoostRegressor'})
    write_page_onnxrt_benches_python(
        None, white_list={
            # 'LGBMClassifier',
            # 'ARDRegression',
            # 'LogisticRegression'
            'HistGradientBoostingRegressor'})
    write_page_onnxrt_benches_onnxruntime1(
        None, white_list={
            # 'LGBMClassifier',
            # 'ARDRegression',
            'HistGradientBoostingRegressor'})
