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

        The command pr

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
            "Unknown format %r." % format)

    if output not in ('', None):
        with open(output, "w", encoding="utf-8") as f:
            f.write(code)
    else:
        fLOG(code)
