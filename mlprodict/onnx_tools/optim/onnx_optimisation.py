"""
@file
@brief Optimisations of :epkg:`ONNX` graphs.
"""
from ..model_checker import check_onnx
from ._onnx_optimisation_common import _apply_optimisation_on_graph
from .onnx_optimisation_identity import onnx_remove_node_identity
from .onnx_optimisation_redundant import onnx_remove_node_redundant
from .onnx_optimisation_unused import onnx_remove_node_unused


def onnx_remove_node(onnx_model, recursive=True, debug_info=None, **options):
    """
    Removes as many nodes as possible without changing
    the outcome. It applies @see fn onnx_remove_node_unused,
    @see fn onnx_remove_node_identity,
    and @see fn onnx_remove_node_redundant.

    @param      onnx_model      onnx model
    @param      recursive       looks into subgraphs
    @param      debug_info      debug information (private)
    @param      options         additional options
    @return                     new onnx model
    """
    if debug_info is None:
        debug_info = [str(type(onnx_model)).rsplit(
            '.', maxsplit=1)[-1].strip("'>")]
    else:
        debug_info = (debug_info +
                      [str(type(onnx_model)).rsplit('.', maxsplit=1)[-1].strip("'>")])

    if hasattr(onnx_model, 'graph'):
        return _apply_optimisation_on_graph(
            onnx_remove_node, onnx_model,
            recursive=recursive, debug_info=debug_info,
            **options)

    check_onnx(onnx_model)
    graph = onnx_model
    graph = onnx_remove_node_unused(
        graph, recursive=recursive, debug_info=debug_info, **options)
    check_onnx(graph)
    graph = onnx_remove_node_identity(
        graph, recursive=recursive, debug_info=debug_info, **options)
    check_onnx(graph)
    graph = onnx_remove_node_redundant(
        graph, recursive=recursive, debug_info=debug_info, **options)
    check_onnx(graph)
    return graph
