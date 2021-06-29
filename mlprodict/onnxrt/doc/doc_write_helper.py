"""
@file
@brief Documentation helper.
"""
from logging import getLogger
from textwrap import indent, dedent
import numpy
from jinja2 import Template
from pandas import DataFrame, notnull
from sklearn.linear_model import LinearRegression
from pyquickhelper.loghelper import noLOG
from pyquickhelper.pandashelper.tblformat import df2rst
from sklearn import __all__ as sklearn__all__
from ...tools.asv_options_helper import get_opset_number_from_onnx
from ...tools.model_info import analyze_model
from ..validate.validate import enumerate_validated_operator_opsets, sklearn_operators
from ...onnx_tools.optim.sklearn_helper import inspect_sklearn_model
from ...onnx_tools.optim.onnx_helper import onnx_statistics
from ..onnx_inference import OnnxInference
from ..validate.validate_summary import _clean_values_optim
from .doc_helper import visual_rst_template


def _make_opset(row):
    opsets = []
    if hasattr(row, 'to_dict'):
        row = row.to_dict()  # pragma: no cover
    for k, v in row.items():
        if k.startswith('opset'):
            if isinstance(v, int):
                opsets.append(v)
            elif isinstance(v, float):  # pragma: no cover
                if numpy.isnan(v):
                    opsets.append(0)
                else:
                    opsets.append(int(v))
            else:  # pragma: no cover
                vv = list(_ for _ in v if 'OK' in str(v))
                if len(vv) > 0:
                    opsets.append(int(k.replace("opset", "")))
    if len(opsets) == 0:
        return "o%d" % get_opset_number_from_onnx()  # pragma: no cover
    val = max(opsets)
    return "o%d" % val


def enumerate_visual_onnx_representation_into_rst(sub, fLOG=noLOG):
    """
    Returns content for pages such as
    :ref:`l-skl2onnx-linear_model`.
    """
    logger = getLogger('skl2onnx')
    logger.disabled = True

    rst_templ = visual_rst_template()
    templ = Template(rst_templ)
    done = set()
    subsets = [_['name'] for _ in sklearn_operators(sub)]
    subsets.sort()
    for row in enumerate_validated_operator_opsets(
            verbose=0, debug=None, fLOG=fLOG,
            opset_min=get_opset_number_from_onnx(),
            opset_max=get_opset_number_from_onnx(),
            store_models=True, models=subsets):

        if 'ONNX' not in row:
            continue
        name = row['name']
        scenario = row['scenario']
        problem = row['problem']
        model = row['MODEL']
        method = row['method_name']
        optim = row.get('optim', '')
        opset = _make_opset(row)
        stats_skl = inspect_sklearn_model(model)
        stats_onx = onnx_statistics(row['ONNX'])
        stats_model = analyze_model(model)
        stats = {'skl_' + k: v for k, v in stats_skl.items()}
        stats.update({'onx_' + k: v for k, v in stats_onx.items()})
        stats.update({'fit_' + k: v for k, v in stats_model.items()})

        df = DataFrame([stats])
        table = df2rst(df.T.reset_index(drop=False))

        clean_optim = _clean_values_optim(optim)
        title = " - ".join([name, problem, scenario, clean_optim])
        if title in done:
            continue  # pragma: no cover
        done.add(title)
        link = "-".join([name, problem, scenario, clean_optim, opset])
        link = link.replace(" ", "").replace(
            "{", "").replace("}", "").replace("'", "")

        optim_param = ("Model was converted with additional parameter: ``{}``.".format(optim)
                       if optim else "")

        oinf = OnnxInference(row['ONNX'], skip_run=True)
        dot = oinf.to_dot(recursive=True)
        try:
            res = templ.render(dot=dot, model=repr(model), method=method,
                               kind=problem, title=title,
                               indent=indent, len=len,
                               link=link, table=table,
                               optim_param=optim_param)
        except KeyError as e:  # pragma: no cover
            rows = [
                '', str(e), '',
                "title='{}'".format(title),
                "method='{}'".format(method),
                "problem='{}'".format(problem),
                model.__class__.__name__, "", "---------",
                rst_templ]
            res = ".. index:: docissue:\n\n::\n\n" + \
                indent("\n".join(rows), "    ")
        yield res


def compose_page_onnxrt_ops(level="^"):
    """
    Writes page :ref:`l-onnx-runtime-operators`.

    @param      level       title level
    """
    begin = dedent("""
    .. _l-onnx-runtime-operators:

    Python Runtime for ONNX operators
    =================================

    The main function instantiates a runtime class which
    computes the outputs of a specific node.

    .. autosignature:: mlprodict.onnxrt.ops.load_op

    Other sections documents available operators.
    This project was mostly started to show a way to
    implement a custom runtime, do some benchmarks,
    test, exepriment...

    .. contents::
        :local:

    Python
    ++++++

    """)
    from ..ops_cpu._op_list import _op_list

    names = []
    for op in _op_list:
        names.append((op.__name__, op))
    names.sort()

    rows = [begin]
    for name, op in names:
        rows.append("")
        rows.append(".. _lpyort-{}:".format(name))
        rows.append("")
        rows.append(name)
        rows.append(level * len(name))
        rows.append("")
        mod = op.__module__.split('.')[-1]
        rows.append(
            ".. autosignature:: mlprodict.onnxrt.ops_cpu.{}.{}".format(mod, name))
        rows.append('')
    return "\n".join(rows)


def split_columns_subsets(df):
    """
    Functions used in the documentation to split
    a dataframe by columns into multiple dataframe to
    reduce the scrolling.
    """
    common = [c for c in ['name', 'problem',
                          'scenario', 'optim'] if c in df.columns]
    subsets = []
    subsets.append(
        [c for c in df.columns if 'opset' in c or 'onx_nnodes' == c])
    subsets.append([c for c in df.columns if 'ERROR' in c or 'opset' in c])
    subsets.append([c for c in df.columns if c.startswith(
        'skl_') or c.startswith('onx_') or 'opset' in c])
    subsets.append([c for c in df.columns if 'N=' in c or 'opset' in c])
    subsets = [s for s in subsets if len(s) > 0]
    return common, subsets


def build_key_split(key, index):
    """
    Used for documentation.
    """
    try:
        new_key = str(key).split('`')[1].split('<')[0].strip()
    except IndexError:
        new_key = str(key)
    if 'SVC' in new_key or 'SVR' in new_key or 'SVM' in new_key:
        return 'SVM'  # pragma: no cover
    if 'Neighbors' in new_key:
        return 'Neighbors'  # pragma: no cover
    if 'Scaler' in new_key:
        return 'Scaler'  # pragma: no cover
    if 'Normalizer' in new_key:
        return 'Scaler'  # pragma: no cover
    if new_key.endswith("NB"):
        return "...NB"  # pragma: no cover
    if new_key.endswith("RBM"):
        return "...NB"  # pragma: no cover
    if "KMeans" in new_key:
        return "KMeans"  # pragma: no cover
    if ('XGB' in new_key or 'LGBM' in new_key or 'Tree' in new_key or
            'Forest' in new_key):
        return 'Trees'  # pragma: no cover
    if ('ARDRegression' in new_key or 'ElasticNet' in new_key or
            'HuberRegressor' in new_key or 'Lars' in new_key or
            'Lasso' in new_key or 'LinearRegression' in new_key or
            'LogisticRegression' in new_key or
            'Ridge' in new_key or 'SGD' in new_key or
            'TheilSen' in new_key):
        return 'Linear'  # pragma: no cover
    for begin in ["Lasso", "Select", "Label", 'Tfidf', 'Feature',
                  'Bernoulli', 'MultiTask', 'OneVs', 'PLS',
                  'Sparse', 'Spectral', 'MiniBatch',
                  'Bayesian']:
        if new_key.startswith(begin):
            return begin + '...'
    for end in ['CV', 'Regressor', 'Classifier']:  # pragma: no cover
        if new_key.endswith(end):
            new_key = new_key[:-len(end)]
    return new_key  # pragma: no cover


def filter_rows(df):
    """
    Used for documentation.
    """
    for c in ['ERROR-msg', 'RT/SKL-N=1']:
        if c in df.columns:
            return df[df[c].apply(lambda x: notnull(x) and x not in (None, '', 'nan'))]
    return df
