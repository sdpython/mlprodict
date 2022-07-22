"""
@file
@brief Command line to check einsum scenarios.
"""


def onnx_code(filename, format="onnx", output=None, verbose=0, name=None,
              opset=None, fLOG=print):
    """
    Exports an ONNX graph into a python code creating
    the same graph.

    :param filename: onnx file
    :param format: format to export too (`onnx`, `tf2onnx`, `numpy`)
    :param output: output file to produce or None to print it on stdout
    :param verbose: verbosity level
    :param name: rewrite the graph name
    :param opset: overwrite the opset (may not works depending on the format)
    :param fLOG: logging function

    .. cmdref::
        :title: Exports an ONNX graph into a python code creating the same graph.
        :cmd: -m mlprodict onnx_code --help
        :lid: l-cmd-onnx_code

        The command converts an ONNX graph into a python code generating
        the same graph. The python code may use onnx syntax, numpy syntax
        or tf2onnx syntax.

        Example::

            python -m mlprodict onnx_code --filename="something.onnx" --format=onnx
    """
    from ..onnx_tools.onnx_export import (  # pylint: disable=E0402
        export2onnx, export2tf2onnx, export2numpy)

    if name == '':
        name = None
    if opset == '':
        opset = None
    try:
        v = int(opset)
        opset = v
    except (ValueError, TypeError):
        opset = None

    if format == 'onnx':
        code = export2onnx(filename, verbose=verbose, name=name, opset=opset)
    elif format == 'tf2onnx':
        code = export2tf2onnx(filename, verbose=verbose,
                              name=name, opset=opset)
    elif format == 'numpy':
        code = export2numpy(filename, verbose=verbose,
                            name=name, opset=opset)
    else:
        raise ValueError(  # pragma: no cover
            f"Unknown format {format!r}.")

    if output not in ('', None):
        with open(output, "w", encoding="utf-8") as f:
            f.write(code)
    else:
        fLOG(code)  # pragma: no cover


def dynamic_doc(verbose=0, fLOG=print):
    """
    Generates the documentation for ONNX operators.

    :param verbose: displays the list of operator
    :param fLOG: logging function
    """
    from ..npy.xop import _dynamic_class_creation
    _dynamic_class_creation(cache=True, verbose=verbose, fLOG=fLOG)


def plot_onnx(filename, format="onnx", verbose=0, output=None, fLOG=print):
    """
    Plots an ONNX graph on the standard output.

    :param filename: onnx file
    :param format: format to export too (`simple`, `tree`, `dot`,
        `io`, `mat`, `raw`)
    :param output: output file to produce or None to print it on stdout
    :param verbose: verbosity level
    :param fLOG: logging function

    .. cmdref::
        :title: Plots an ONNX graph as text
        :cmd: -m mlprodict plot_onnx --help
        :lid: l-cmd-plot_onnx

        The command shows the ONNX graphs as a text on the standard output.

        Example::

            python -m mlprodict plot_onnx --filename="something.onnx" --format=simple
    """
    if isinstance(filename, str):
        from onnx import load
        content = load(filename)
    else:
        content = filename
    if format == 'dot':
        from ..onnxrt import OnnxInference
        code = OnnxInference(filename).to_dot()
    elif format == 'simple':
        from mlprodict.plotting.text_plot import onnx_simple_text_plot
        code = onnx_simple_text_plot(content)
    elif format == 'io':
        from mlprodict.plotting.text_plot import onnx_text_plot_io
        code = onnx_text_plot_io(content)
    elif format == 'mat':
        from mlprodict.plotting.text_plot import onnx_text_plot
        code = onnx_text_plot(content)
    elif format == 'raw':
        code = str(content)
    elif format == 'tree':
        from mlprodict.plotting.plotting import onnx_text_plot_tree
        rows = []
        for node in content.graph.node:
            if node.op_type.startswith("TreeEnsemble"):
                rows.append(f'Node type={node.op_type!r} name={node.name!r}')
                rows.append(onnx_text_plot_tree(node))
        code = "\n".join(rows)
    else:
        raise ValueError(  # pragma: no cover
            f"Unknown format {format!r}.")

    if output not in ('', None):
        with open(output, "w", encoding="utf-8") as f:
            f.write(code)
    else:
        fLOG(code)  # pragma: no cover
