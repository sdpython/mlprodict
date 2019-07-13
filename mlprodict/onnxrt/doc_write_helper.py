"""
@file
@brief Documentation helper.
"""
from logging import getLogger
from textwrap import indent, dedent
from jinja2 import Template
from sklearn.linear_model import LinearRegression
from pyquickhelper.loghelper import noLOG
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, sklearn_operators
from mlprodict.onnxrt.validate import get_opset_number_from_onnx, sklearn__all__
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.doc_helper import visual_rst_template


def enumerate_visual_onnx_representation_into_rst(sub, fLOG=noLOG):
    """
    Returns content for pages such as
    :ref:`l-skl2onnx-linear_model`.
    """
    logger = getLogger('skl2onnx')
    logger.disabled = True

    templ = Template(visual_rst_template())
    done = set()
    subsets = [_['name'] for _ in sklearn_operators(sub)]
    subsets.sort()
    for row in enumerate_validated_operator_opsets(
            verbose=0, debug=None, fLOG=fLOG, opset_min=get_opset_number_from_onnx(),
            store_models=True, models=subsets):

        if 'ONNX' not in row:
            continue
        name = row['name']
        scenario = row['scenario']
        problem = row['problem']
        model = row['MODEL']
        method = row['method_name']
        title = " - ".join([name, problem, scenario])
        if title in done:
            continue
        done.add(title)
        link = "-".join([name, problem, scenario])

        oinf = OnnxInference(row['ONNX'], skip_run=True)
        dot = oinf.to_dot()
        res = templ.render(dot=dot, model=repr(model), method=method,
                           kind=problem, title=title,
                           indent=indent, len=len,
                           link=link)
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
    from .ops_cpu._op_list import _op_list

    names = []
    for op in _op_list:
        names.append((op.__name__, op))
    names.sort()

    rows = [begin]
    for name, op in names:
        rows.append(name)
        rows.append(level * len(name))
        rows.append("")
        mod = op.__module__.split('.')[-1]
        rows.append(
            ".. autosignature:: mlprodict.onnxrt.ops_cpu.{}.{}".format(mod, name))
        rows.append('')
    return "\n".join(rows)
