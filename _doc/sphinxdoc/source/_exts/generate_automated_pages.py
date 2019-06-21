"""
Extensions for mlprodict.
"""
import os
from textwrap import dedent
from logging import getLogger
from pandas import DataFrame
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
import sphinx
from pyquickhelper.loghelper import noLOG
from pyquickhelper.pycode import is_travis_or_appveyor
from pyquickhelper.pandashelper import df2rst


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

    def make_link(row):
        link = "`{name} <l-{name}-{problem}-{scenario}>`"
        name = row['name']
        problem = row['problem']
        scenario = row['scenario']
        return link.format(name=name, problem=problem,
                           scenario=scenario)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning,
                               RuntimeWarning, FutureWarning))
    def build_table():
        logger = getLogger('skl2onnx')
        logger.disabled = True
        benchmark = is_travis_or_appveyor() != 'circleci'
        rows = list(enumerate_validated_operator_opsets(11, debug=None, fLOG=print,
                                                        runtime=runtime,
                                                        benchmark=benchmark))
        df = DataFrame(rows)
        piv = summary_report(df)
        piv['name'] = piv.apply(lambda row: make_link(row), axis=1)

        if "ERROR-msg" in piv.columns:
            def shorten(text):
                text = str(text)
                if len(text) > 75:
                    text = text[:75] + "..."
                return text

            piv["ERROR-msg"] = piv["ERROR-msg"].apply(shorten)

        return df2rst(piv)

    with open(whe, 'w', encoding='utf-8') as f:
        title = "Available of scikit-learn model for runtime {0}".format(
            runtime)
        f.write(dedent('''
        _l-onnx-bench-{0}:

        {1}
        {2}

        The following metrics measure the ratio between the prediction time
        for the runtime compare to :epkg:`scikit-learn`.
        It gives an order of magnitude.

        '''.format(runtime, title, "=" * len(title))))
        f.write(build_table())
    logger.info(
        "[mlprodict] done page '{}'.".format(whe))
    print("[mlprodict-sphinx] done page runtime '{}' - '{}'.".format(runtime, whe))


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
