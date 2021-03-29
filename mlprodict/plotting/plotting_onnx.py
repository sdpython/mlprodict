"""
@file
@brief Useful plots.
"""
import matplotlib.pyplot as plt
from ..onnxrt import OnnxInference


def plot_onnx(onx, ax=None, dpi=300, temp_dot=None, temp_img=None,
              show=False):
    """
    Plots an ONNX graph into graph.

    :param onx: ONNX object, @see cl OnnxInference
    :param ax: existing axes
    :param dpi: resolution
    :param temp_dot: temporary file,
        if None, a file is created and removed
    :param temp_img: temporary image,
        if None, a file is created and removed
    :param show: calls `plt.show()`
    :return: axes
    """
    # delayed import
    from pyquickhelper.helpgen.graphviz_helper import plot_graphviz

    if ax is None:
        ax = plt.gca()  # pragma: no cover
    if not isinstance(onx, OnnxInference):
        oinf = OnnxInference(onx, skip_run=True)
    else:
        oinf = onx  # pragma: no cover
    dot = oinf.to_dot()
    plot_graphviz(dot, dpi=dpi, ax=ax, temp_dot=temp_dot, temp_img=temp_img)
    if show:
        plt.show()  # pragma: no cover
    return ax
