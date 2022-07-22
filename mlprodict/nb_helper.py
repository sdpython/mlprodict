"""
@file
@brief Helpers for notebooks.
"""
from IPython.core.magic import magics_class, line_magic
from jyquickhelper import RenderJsDot
from pyquickhelper.ipythonhelper import MagicCommandParser, MagicClassWithHelpers
from pyquickhelper.cli.cli_helper import create_cli_parser


def onnxview(graph, recursive=False, local=False, add_rt_shapes=False,
             runtime='python', size=None, html_size=None):
    """
    Displays an :epkg:`ONNX` graph into a notebook.

    :param graph: filename, bytes, or :epkg:`onnx` graph.
    :param recursive: display subgraph
    :param local: use local path to javascript dependencies,
        recommanded option if used on :epkg:`MyBinder`)
    :param add_rt_shapes: add information about the shapes
        the runtime was able to find out,
        the runtime has to be `'python'`
    :param runtime: the view fails if a runtime does not implement a specific
        node unless *runtime* is `'empty'`
    :param size: graph size
    :param html_size: html size

    .. versionchanged:: 0.6
        Parameter *runtime* was added.
    """
    from .onnxrt import OnnxInference
    sess = OnnxInference(graph, skip_run=not add_rt_shapes, runtime=runtime)
    dot = sess.to_dot(recursive=recursive,
                      add_rt_shapes=add_rt_shapes, size=size)
    if html_size is not None:
        return RenderJsDot(dot, local=local, width=html_size, height=html_size)
    return RenderJsDot(dot, local=local)  # pragma: no cover


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

            The magic command ``%onnxview model_onnx`` is equivalent to function
            :func:`onnxview <mlprodict.onnxrt.doc.nb_helper.onnxview>`:

            ::

                onnx_view(model_onnx)

            It displays a visual representation of an :epkg:`ONNX` graph.

        """
        parser = self.get_parser(
            lambda: create_cli_parser(onnxview, cls=MagicCommandParser,
                                      positional={'graph'}), "onnxview")
        args = self.get_args(line, parser)

        if args is not None:
            size = args.size
            if size == "":
                size = None
            res = onnxview(args.graph, recursive=args.recursive,
                           local=args.local, add_rt_shapes=args.add_rt_shapes,
                           size=size, html_size=args.html_size)
            return res
        return None


def register_onnx_magics(ip=None):  # pragma: no cover
    """
    Register magics function, can be called from a notebook.

    @param      ip      from ``get_ipython()``
    """
    if ip is None:
        from IPython import get_ipython
        ip = get_ipython()
    ip.register_magics(OnnxNotebook)
