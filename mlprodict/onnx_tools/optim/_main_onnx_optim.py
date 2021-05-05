"""
@file
@brief Calls all possible :epkg:`ONNX` optimisations.
"""
from .onnx_optimisation import onnx_remove_node


def onnx_optimisations(onnx_model, recursive=True, debug_info=None, **options):
    """
    Calls several possible optimisations including
    @see fn onnx_remove_node.

    @param      onnx_model      onnx model
    @param      recursive       looks into subgraphs
    @param      debug_info      debug information (private)
    @param      options         additional options
    @return                     new onnx _model
    """
    new_model = onnx_remove_node(
        onnx_model, recursive=recursive, debug_info=debug_info,
        **options)
    return new_model
