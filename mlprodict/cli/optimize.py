"""
@file
@brief Command line about model optimisation.
"""
import os
import onnx


def onnx_stats(name, optim=False, kind=None):
    """
    Computes statistics on an ONNX model.

    :param name: filename
    :param optim: computes statistics before an after optimisation was done
    :param kind: kind of statistics, if left unknown,
        prints out the metadata, possible values:
        * `io`: prints input and output name, type, shapes
        * `node`: prints the distribution of node types
        * `text`: printts a text summary

    .. cmdref::
        :title: Computes statistics on an ONNX graph
        :cmd: -m mlprodict onnx_stats --help
        :lid: l-cmd-onnx_stats

        The command computes statistics on an ONNX model.
    """
    if not os.path.exists(name):
        raise FileNotFoundError(  # pragma: no cover
            f"Unable to find file '{name}'.")
    with open(name, 'rb') as f:
        model = onnx.load(f)
    if kind in (None, ""):
        from ..onnx_tools.optim import onnx_statistics
        return onnx_statistics(model, optim=optim)
    if kind == 'text':
        from ..plotting.plotting import onnx_simple_text_plot
        return onnx_simple_text_plot(model)
    if kind == 'io':
        from ..plotting.plotting import onnx_text_plot_io
        return onnx_text_plot_io(model)
    if kind == 'node':
        from ..onnx_tools.optim import onnx_statistics
        return onnx_statistics(model, optim=optim, node_type=True)
    raise ValueError(  # pragma: no cover
        f"Unexpected kind={kind!r}.")


def onnx_optim(name, outfile=None, recursive=True, options=None, verbose=0, fLOG=None):
    """
    Optimizes an ONNX model.

    :param name: filename
    :param outfile: output filename
    :param recursive: processes the main graph and the subgraphs
    :param options: options, kind of optimize to do
    :param verbose: display statistics before and after the optimisation
    :param fLOG: logging function

    .. cmdref::
        :title: Optimizes an ONNX graph
        :cmd: -m mlprodict onnx_optim --help
        :lid: l-cmd-onnx_optim

        The command optimizes an ONNX model.
    """
    from ..onnx_tools.optim import onnx_statistics, onnx_optimisations
    if not os.path.exists(name):
        raise FileNotFoundError(  # pragma: no cover
            f"Unable to find file '{name}'.")
    if outfile == "":
        outfile = None  # pragma: no cover
    if options == "":
        options = None  # pragma: no cover
    if verbose >= 1 and fLOG is not None:
        fLOG(f"[onnx_optim] read file '{name}'.")
    with open(name, 'rb') as f:
        model = onnx.load(f)
    if verbose >= 1 and fLOG is not None:
        stats = onnx_statistics(model, optim=False)
        for k, v in sorted(stats.items()):
            fLOG(f'  before.{k}={v}')
    new_model = onnx_optimisations(model, recursive=recursive)
    if verbose >= 1 and fLOG is not None:
        stats = onnx_statistics(model, optim=False)
        for k, v in sorted(stats.items()):
            fLOG(f'   after.{k}={v}')
    if outfile is not None:
        fLOG(f"[onnx_optim] write '{outfile}'.")
        with open(outfile, 'wb') as f:
            onnx.save(new_model, f)
    return new_model
