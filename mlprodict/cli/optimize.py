"""
@file
@brief Command line about model optimisation.
"""
import os
import onnx


def onnx_stats(name, optim=False):
    """
    Computes statistics on an ONNX model.

    :param name: filename
    :param optim: computes statistics before an after optimisation was done

    .. cmdref::
        :title: Computes statistics on an ONNX graph
        :cmd: -m mlprodict onnx_stats --help
        :lid: l-cmd-onnx_stats

        The command computes statistics on an ONNX model.
    """
    from ..onnxrt.optim import onnx_statistics
    if not os.path.exists(name):
        raise FileNotFoundError(  # pragma: no cover
            "Unable to find file '{}'.".format(name))
    with open(name, 'rb') as f:
        model = onnx.load(f)
    return onnx_statistics(model, optim=optim)


def onnx_optim(name, outfile=None, recursive=True, options=None, verbose=0, fLOG=None):
    """
    Optimises an ONNX model.

    :param name: filename
    :param outfile: output filename
    :param recursive: processes the main graph and the subgraphs
    :param options: options, kind of optimize to do
    :param verbose: display statistics before and after the optimisation
    :param fLOG: logging function

    .. cmdref::
        :title: Optimises an ONNX graph
        :cmd: -m mlprodict onnx_optim --help
        :lid: l-cmd-onnx_optim

        The command optimises an ONNX model.
    """
    from ..onnxrt.optim import onnx_statistics, onnx_optimisations
    if not os.path.exists(name):
        raise FileNotFoundError(  # pragma: no cover
            "Unable to find file '{}'.".format(name))
    if outfile == "":
        outfile = None  # pragma: no cover
    if options == "":
        options = None  # pragma: no cover
    if verbose >= 1 and fLOG is not None:
        fLOG("[onnx_optim] read file '{}'.".format(name))
    with open(name, 'rb') as f:
        model = onnx.load(f)
    if verbose >= 1 and fLOG is not None:
        stats = onnx_statistics(model, optim=False)
        for k, v in sorted(stats.items()):
            fLOG('  before.{}={}'.format(k, v))
    new_model = onnx_optimisations(model, recursive=recursive)
    if verbose >= 1 and fLOG is not None:
        stats = onnx_statistics(model, optim=False)
        for k, v in sorted(stats.items()):
            fLOG('   after.{}={}'.format(k, v))
    if outfile is not None:
        fLOG("[onnx_optim] write '{}'.".format(outfile))
        with open(outfile, 'wb') as f:
            onnx.save(new_model, f)
    return new_model
