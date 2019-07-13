"""
Extensions for mlprodict.
"""
import os
from textwrap import dedent
from logging import getLogger
from pandas import DataFrame, read_excel
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
import sphinx
from pyquickhelper.loghelper import noLOG
from pyquickhelper.pycode import is_travis_or_appveyor
from pyquickhelper.pandashelper import df2rst
from pyquickhelper.loghelper import run_cmd
from pyquickhelper.loghelper.run_cmd import get_interpreter_path


@ignore_warnings(category=(UserWarning, ConvergenceWarning,
                           RuntimeWarning, FutureWarning))
def write_page_onnxrt_ops(app):
    from mlprodict.onnxrt.doc_write_helper import compose_page_onnxrt_ops
    logger = getLogger('mlprodict')
    srcdir = app.builder.srcdir
    whe = os.path.join(os.path.abspath(srcdir), "api", "onnxrt_ops.rst")
    logger.info("[mlprodict] create page '{}'.".format(whe))
    print("[mlprodict-sphinx] create page '{}'.".format(whe))
    page = compose_page_onnxrt_ops()
    with open(whe, "w", encoding='utf-8') as f:
        f.write(page)
    print("[mlprodict-sphinx] done page '{}'.".format(whe))


def write_page_onnxrt_benches(app, runtime, skip=None):

    from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report
    logger = getLogger('mlprodict')
    srcdir = app.builder.srcdir

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

    out_raw = os.path.join(srcdir, "bench_raw_%s.xlsx" % runtime)
    out_sum = os.path.join(srcdir, "bench_sum_%s.xlsx" % runtime)
    cmd = ('{0} -m mlprodict validate_runtime --verbose=1 --out_raw={1} --out_summary={2} '
           '--benchmark=1 --dump_folder={3} --runtime={4}{5}'.format(
               get_interpreter_path(), out_raw, out_sum, srcdir, runtime,
               " --skip_models={}".format(','.join(skip)) if skip else "")
    logger.info("[mlprodict] cmd '{}'.".format(cmd))
    print("[mlprodict-sphinx] cmd '{}'".format(cmd))
    out, err=run_cmd(cmd, wait=True, fLOG=print)

    logger.info("[mlprodict] reading '{}'.".format(out_sum))
    print("[mlprodict-sphinx] reading '{}'".format(out_sum))

    if os.path.exists(out_sum):
        piv=read_excel(out_sum)

        logger.info("[mlprodict] shape '{}'.".format(piv.shape))
        print("[mlprodict-sphinx] shape '{}'".format(piv.shape))

        def make_link(row):
            link=":ref:`{name} <l-{name}-{problem}-{scenario}>`"
            name=row['name']
            problem=row['problem']
            scenario=row['scenario']
            return link.format(name=name, problem=problem,
                               scenario=scenario)

        piv['name']=piv.apply(lambda row: make_link(row), axis=1)

        if "ERROR-msg" in piv.columns:
            def shorten(text):
                text=str(text)
                if len(text) > 75:
                    text=text[:75] + "..."
                return text

            piv["ERROR-msg"]=piv["ERROR-msg"].apply(shorten)

        logger.info("[mlprodict] write '{}'.".format(whe))
        print("[mlprodict-sphinx] write '{}'".format(whe))

        with open(whe, 'w', encoding='utf-8') as f:
            title="Available of scikit-learn model for runtime {0}".format(
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

            Some figures are missing when the number of observations is high.
            That means the prediction is slow for one of the runtime
            (ONNX, scikit-learn) and it would take too long to go further.

            '''.format(runtime, title, "=" * len(title))))
            f.write(df2rst(piv, number_format=2,
                           replacements={'nan': '', 'ERR: 4convert': ''}))
        logger.info(
            "[mlprodict] done page '{}'.".format(whe))
        print("[mlprodict-sphinx] done page runtime '{}' - '{}'.".format(runtime, whe))

    else:
        logger.warning("[mlprodict] unable to find '{}'.".format(out_sum))
        print("[mlprodict-sphinx] unable to find '{}'".format(out_sum))
        raise RuntimeError("--OUT--\n{}\n--ERR--\n{}".format(out, err))


def write_page_onnxrt_benches_python(app):
    write_page_onnxrt_benches(app, 'python')


def write_page_onnxrt_benches_onnxruntime2(app):
    write_page_onnxrt_benches(app, 'onnxruntime2',
        {AdaBoostRegressor, GaussianProcessClassifier})


def write_page_onnxrt_benches_onnxruntime1(app):
    write_page_onnxrt_benches(app, 'onnxruntime1')


def setup(app):
    """
    Preparation of the documentation.
    """
    app.connect('builder-inited', write_page_onnxrt_benches_python)
    app.connect('builder-inited', write_page_onnxrt_benches_onnxruntime1)
    app.connect('builder-inited', write_page_onnxrt_benches_onnxruntime2)
    app.connect('builder-inited', write_page_onnxrt_ops)
    return {'version': sphinx.__display_version__,
            'parallel_read_safe': False,
            'parallel_write_safe': False}
