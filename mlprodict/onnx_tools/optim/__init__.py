"""
@file
@brief Shortcuts to *onnx_tools.optim*.
"""
from .onnx_helper import onnx_statistics
from .onnx_optimisation_identity import onnx_remove_node_identity
from .onnx_optimisation_redundant import onnx_remove_node_redundant
from .onnx_optimisation_unused import onnx_remove_node_unused
from .onnx_optimisation import onnx_remove_node
from ._main_onnx_optim import onnx_optimisations
