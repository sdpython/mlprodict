"""
@file
@brief Command line about model manipulations.
"""


def replace_initializer(filename, output=None, verbose=0, threshold=128,
                        rename=False, fLOG=print):
    """
    Replaces big initializers by node *ConstantOfShape* to
    help having lighter unit tests.

    :param filename: onnx file
    :param output: output file to produce or None to print it on stdout
    :param verbose: verbosity level
    :param rename: rename names to reduce name footprint
    :param threshold: replace all initializer above that size
    :param fLOG: logging function

    .. cmdref::
        :title: Replaces big initializers by node *ConstantOfShape*
        :cmd: -m mlprodict replace_initializer --help
        :lid: l-cmd-replace_initializer

        The command replaces big initializers by node *ConstantOfShape* to
    help having lighter unit tests.

        Example::

            python -m mlprodict replace_initializer --filename="something.onnx" --output="modified.onnx"
    """
    from onnx import load
    from onnx.checker import check_model
    from onnx.onnx_cpp2py_export.checker import ValidationError  # pylint: disable=E0611, E0401
    from ..onnx_tools.onnx_manipulations import (  # pylint: disable=E0402
        replace_initializer_by_constant_of_shape,
        onnx_rename_names)

    if filename == '':
        filename = None  # pragma: no cover
    if threshold:
        threshold = int(threshold)
    if rename:
        rename = rename in (1, '1', 'true', 'True', True)

    with open(filename, "rb") as f:
        onx = load(f)
    if rename:
        onx = onnx_rename_names(onx)
    new_onx = replace_initializer_by_constant_of_shape(
        onx, threshold=threshold)
    try:
        check_model(new_onx)
    except ValidationError as e:
        if output not in ('', None):
            with open(output + ".error.onnx", "wb") as f:
                f.write(new_onx.SerializeToString())
        raise e

    if output not in ('', None):
        with open(output, "wb") as f:
            f.write(new_onx.SerializeToString())
    else:
        fLOG(new_onnx)  # pragma: no cover
