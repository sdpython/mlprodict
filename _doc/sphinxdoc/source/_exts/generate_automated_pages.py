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


def write_page_onnxrt_benches(app, runtime):

    from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report
    logger = getLogger('mlprodict')
    srcdir = app.builder.srcdir

    if runtime == 'CPU':
        whe = os.path.join(os.path.abspath(srcdir),
                           "skl_converters", "bench_python.rst")
    elif runtime == 'onnxruntime':
        whe = os.path.join(os.path.abspath(srcdir),
                           "skl_converters", "bench_onnxrt.rst")
    elif runtime == 'onnxruntime-whole':
        whe = os.path.join(os.path.abspath(srcdir),
                           "skl_converters", "bench_onnxrt_whole.rst")
    else:
        raise RuntimeError("Unsupported runtime '{}'.".format(runtime))

    logger.info("[mlprodict] create page '{}'.".format(whe))
    print("[mlprodict-sphinx] create page runtime '{}' - '{}'.".format(runtime, whe))

    out_raw = os.path.join(srcdir, "bench_raw_%s.xlsx" % runtime)
    out_sum = os.path.join(srcdir, "bench_sum_%s.xlsx" % runtime)
    cmd = ('{0} -m mlprodict validate_runtime --verbose=1 --out_raw={1} --out_summary={2} '
           '--benchmark=1 --dump_folder={3} --runtime={4}'.format(
               get_interpreter_path(), out_raw, out_sum, srcdir, runtime))
    logger.info("[mlprodict] cmd '{}'.".format(cmd))
    print("[mlprodict-sphinx] cmd '{}'".format(cmd))
    out, err = run_cmd(cmd, wait=True, fLOG=print)

    logger.info("[mlprodict] reading '{}'.".format(out_sum))
    print("[mlprodict-sphinx] reading '{}'".format(out_sum))

    if os.path.exists(out_sum):
        piv = read_excel(out_sum)

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
            title = "Available of scikit-learn model for runtime {0}".format(
                runtime)
            f.write(dedent('''
            .. _l-onnx-bench-{0}:

            {1}
            {2}

            The following metrics measure the ratio between the prediction time
            for the runtime compare to :epkg:`scikit-learn`.
            It gives an order of magnitude.

            '''.format(runtime, title, "=" * len(title))))
            f.write(df2rst(piv, number_format=2))
        logger.info(
            "[mlprodict] done page '{}'.".format(whe))
        print("[mlprodict-sphinx] done page runtime '{}' - '{}'.".format(runtime, whe))

    else:
        logger.warning("[mlprodict] unable to find '{}'.".format(out_sum))
        print("[mlprodict-sphinx] unable to find '{}'".format(out_sum))
        raise RuntimeError("--OUT--\n{}\n--ERR--\n{}".format(out, err))


def write_page_onnxrt_benches_cpu(app):
    write_page_onnxrt_benches(app, 'CPU')


def write_page_onnxrt_benches_onnxruntime(app):
    write_page_onnxrt_benches(app, 'onnxruntime')


def write_page_onnxrt_benches_onnxruntime_whole(app):
    write_page_onnxrt_benches(app, 'onnxruntime-whole')


def setup(app):
    """
    Preparation of the documentation.
    """
    app.connect('builder-inited', write_page_onnxrt_benches_cpu)
    app.connect('builder-inited', write_page_onnxrt_benches_onnxruntime)
    app.connect('builder-inited', write_page_onnxrt_benches_onnxruntime_whole)
    app.connect('builder-inited', write_page_onnxrt_ops)
    return {'version': sphinx.__display_version__,
            'parallel_read_safe': False,
            'parallel_write_safe': False}
