
Tools for ONNX
==============

.. contents::
    :local:

Accessor
++++++++

.. autosignature:: mlprodict.onnx_tools.onnx_tools.find_node_input_name

.. autosignature:: mlprodict.onnx_tools.onnx_tools.find_node_name

.. autosignature:: mlprodict.onnx_tools.onnx_tools.insert_node

Optimisation
++++++++++++

The following functions reduce the number of ONNX operators in a graph
while keeping the same results. The optimized graph
is left unchanged.

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation.onnx_remove_node

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation_identity.onnx_remove_node_identity

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation_redundant.onnx_remove_node_redundant

.. autosignature:: mlprodict.onnx_tools.optim.onnx_remove_unused.onnx_remove_node_unused

Validation
++++++++++

.. autosignature:: mlprodict.onnx_tools.model_checker.onnx_shaker

