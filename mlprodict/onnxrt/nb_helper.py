"""
@file
@brief Helpers for notebooks.
"""
from .onnx_inference import OnnxInference  # pylint: disable=C0411
from IPython.core.magic import magics_class, line_magic  # pylint: disable=C0411
from jyquickhelper import RenderJsDot  # pylint: disable=C0411
from pyquickhelper.ipythonhelper import MagicCommandParser, MagicClassWithHelpers  # pylint: disable=C0411
from pyquickhelper.cli.cli_helper import create_cli_parser  # pylint: disable=C0411


def onnxview(graph):
    """
    Displays an :epkg:`ONNX` graph into a notebook.

    :param graph:  filename, bytes, or :epkg:`onnx` graph.
    """
    sess = OnnxInference(graph, skip_run=True)
    dot = sess.to_dot()
    return RenderJsDot(dot)


@magics_class
class OnnxNotebook(MagicClassWithHelpers):

    """
    Defines magic commands to help with notebooks

    .. versionadded:: 1.1
    """

    @line_magic
    def onnxview(self, line):
        """
        Defines ``%onnxview``
        which displays an :epkg:`ONNX` graph.

        .. nbref::
            :title: onnxview

            The magic command ``%onnxview`` is equivalent to::

                onnx_view(model_onnx)
        """
        parser = self.get_parser(lambda: create_cli_parser(onnxview, cls=MagicCommandParser,
                                                           positional={'graph'}), "onnxview")
        args = self.get_args(line, parser)

        if args is not None:
            res = onnxview(args.graph)
            return res
        return None


def register_onnx_magics(ip=None):
    """
    register magics function, can be called from a notebook

    @param      ip      from ``get_ipython()``
    """
    if ip is None:
        from IPython import get_ipython
        ip = get_ipython()
    ip.register_magics(OnnxNotebook)
