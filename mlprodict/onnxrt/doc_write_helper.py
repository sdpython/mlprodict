"""
@file
@brief Documentation helper.
"""
from logging import getLogger
from textwrap import indent
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
        method = row['method']
        title = " - ".join([name, scenario])
        if title in done:
            continue
        done.add(title)

        oinf = OnnxInference(row['ONNX'], skip_run=True)
        dot = oinf.to_dot()
        res = templ.render(dot=dot, model=repr(model), method=method,
                           kind=problem, title=title,
                           indent=indent, len=len)
        yield res
