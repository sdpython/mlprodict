"""
Extensions for mlprodict.
"""
import os
from textwrap import dedent
from logging import getLogger
from pandas import DataFrame, read_excel, read_csv, concat, Series, notnull
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
import sphinx
from tqdm import tqdm
from pyquickhelper.loghelper import noLOG
from pyquickhelper.pycode import is_travis_or_appveyor
from pyquickhelper.pandashelper import df2rst
from pyquickhelper.loghelper import run_cmd
from pyquickhelper.loghelper.run_cmd import get_interpreter_path
from mlprodict.onnxrt.validate.validate_helper import sklearn_operators
from mlprodict.onnxrt.doc.doc_write_helper import split_columns_subsets


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


def run_benchmark(runtime, srcdir, logger, skip, white_list=None):
    filenames = []
    skls = sklearn_operators(extended=True)
    skls = [_['name'] for _ in skls]
    if white_list:
        skls = [_ for _ in skls if _ in white_list]
    skls.sort()
    pbar = tqdm(skls)
    for op in pbar:
        if skip is not None and op in skip:
            continue
        pbar.set_description("[%s]" % (op + " " * (25 - len(op))))

        out_raw = os.path.join(srcdir, "bench_raw_%s_%s.csv" % (runtime, op))
        out_sum = os.path.join(srcdir, "bench_sum_%s_%s.csv" % (runtime, op))
        cmd = ('{0} -m mlprodict validate_runtime --verbose=0 --out_raw={1} --out_summary={2} '
               '--benchmark=1 --dump_folder={3} --runtime={4} --models={5}'.format(
                   get_interpreter_path(), out_raw, out_sum, srcdir, runtime, op))
        logger.info("[mlprodict] cmd '{}'.".format(cmd))
        out, err = run_cmd(cmd, wait=True, fLOG=None)
        if not os.path.exists(out_sum):
            logger.warning("[mlprodict] unable to find '{}'.".format(out_sum))
            print("[mlprodict-sphinx] cmd '{}'".format(cmd))
            print("[mlprodict-sphinx] unable to find '{}'".format(out_sum))
            msg = "Unable to find '{}'\n--CMD--\n{}\n--OUT--\n{}\n--ERR--\n{}".format(
                out_sum, cmd, out, err)
            print(msg)
            rows = [{'name': op, 'scenario': 'CRASH',
                     'ERROR-msg': msg.replace("\n", " -- ")}]
            df = DataFrame(rows)
            df.to_csv(out_sum, index=False)
        filenames.append((out_raw, out_sum))
    return filenames


def write_page_onnxrt_benches(app, runtime, skip=None, white_list=None):

    from mlprodict.onnxrt.validate.validate import enumerate_validated_operator_opsets
    logger = getLogger('mlprodict')
    srcdir = app.builder.srcdir if app is not None else ".."

    if runtime in ('python'):
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

    filenames = run_benchmark(runtime, srcdir, logger, skip,
                              white_list=white_list)
    dfs_raw = [read_csv(name[0]) for name in filenames]
    dfs_sum = [read_csv(name[1]) for name in filenames]
    df_raw = concat(dfs_raw, sort=False)
    piv = concat(dfs_sum, sort=False)

    opset_cols = [(int(oc.replace("opset", "")), oc)
                  for oc in piv.columns if 'opset' in oc]
    opset_cols.sort(reverse=True)
    opset_cols = [oc[1] for oc in opset_cols]
    new_cols = opset_cols[:1]
    bench_cols = ["RT/SKL-N=1", "N=10", "N=100",
                  "N=1000", "N=10000", "N=100000"]
    new_cols.extend(["ERROR-msg", "name", "problem", "scenario"])
    new_cols.extend(bench_cols)
    new_cols.extend(opset_cols[1:])
    for c in bench_cols:
        new_cols.append(c + '-min')
        new_cols.append(c + '-max')
    new_cols = [_ for _ in new_cols if _ in piv.columns]
    piv = piv[new_cols]

    out_sum = os.path.join(srcdir, "bench_sum_%s.xlsx" % runtime)
    piv.to_excel(out_sum, index=False)
    logger.info("[mlprodict] wrote '{}'.".format(out_sum))
    print("[mlprodict-sphinx] wrote '{}'".format(out_sum))

    out_raw = os.path.join(srcdir, "bench_raw_%s.xlsx" % runtime)
    df_raw.to_excel(out_raw, index=False)
    logger.info("[mlprodict] wrote '{}'.".format(out_raw))
    print("[mlprodict-sphinx] wrote '{}'".format(out_raw))

    logger.info("[mlprodict] shape '{}'.".format(piv.shape))
    print("[mlprodict-sphinx] shape '{}'".format(piv.shape))

    def make_link(row):
        link = ":ref:`{name} <l-{name}-{problem}-{scenario}>`"
        name = row['name']
        problem = row['problem']
        scenario = row['scenario']
        return link.format(name=name, problem=problem,
                           scenario=scenario)

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

    def build_key_split(key, index):
        try:
            new_key = str(key).split('`')[1].split('<')[0].strip()
        except IndexError:
            new_key = str(key)
        return new_key

    def filter_rows(df):
        for c in ['ERROR-msg', 'RT/SKL-N=1']:
            if c in df.columns:
                return df[df[c].apply(lambda x: notnull(x) and x not in (None, '', 'nan'))]
        return df

    with open(whe, 'w', encoding='utf-8') as f:
        title = "Available of scikit-learn model for runtime {0}".format(
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
                       filter_rows=filter_rows))
    logger.info(
        "[mlprodict] done page '{}'.".format(whe))
    print("[mlprodict-sphinx] done page runtime '{}' - '{}'.".format(runtime, whe))


def write_page_onnxrt_benches_python(app, white_list=None):
    write_page_onnxrt_benches(app, 'python', white_list=white_list)


def write_page_onnxrt_benches_onnxruntime2(app, white_list=None):
    write_page_onnxrt_benches(
        app, 'onnxruntime2',
        {AdaBoostRegressor, GaussianProcessClassifier},
        white_list=white_list)


def write_page_onnxrt_benches_onnxruntime1(app, white_list=None):
    write_page_onnxrt_benches(app, 'onnxruntime1', white_list=white_list)


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
    write_page_onnxrt_benches_onnxruntime1(
        None, white_list={'LGBMClassifier', 'ARDRegression'})
