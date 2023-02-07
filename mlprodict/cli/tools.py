"""
@file
@brief Command line about model manipulations.
"""


def replace_initializer(filename, output=None, verbose=0, threshold=128,
                        fLOG=print):
    """
    Replaces big initializers by node *ConstantOfShape* to
    help having lighter unit tests.

    :param filename: onnx file
    :param output: output file to produce or None to print it on stdout
    :param verbose: verbosity level
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
    from ..onnx_tools.onnx_manipulations import (  # pylint: disable=E0402
        replace_initializer_by_constant_of_shape)

    if name == '':
        name = None  # pragma: no cover

    with open(filename, "rb") as f:
        onx = onx.load(f)
    new_onx = replace_initializer_by_constant_of_shape(
        onx, threshold=threshold)
    if output not in ('', None):
        with open(output, "wb") as f:
            f.write(new_onx.SerializeToString())
    else:
        fLOG(code)  # pragma: no cover
