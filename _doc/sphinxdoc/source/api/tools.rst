
=====
Tools
=====

.. contents::
    :local:

ONNX
====

Accessor
++++++++

.. autosignature:: mlprodict.onnx_tools.onnx_tools.find_node_input_name

.. autosignature:: mlprodict.onnx_tools.onnx_tools.find_node_name

.. autosignature:: mlprodict.onnx_tools.onnx_tools.insert_node

Export
++++++

.. autosignature:: mlprodict.onnx_tools.onnx_export.export2numpy

.. autosignature:: mlprodict.onnx_tools.onnx_export.export2onnx

.. autosignature:: mlprodict.onnx_tools.onnx_export.export2tf2onnx

Graphs
++++++

Functions to help understand models.

.. autosignature:: mlprodict.tools.model_info.analyze_model

.. autosignature:: mlprodict.onnx_tools.onnx_manipulations.enumerate_model_node_outputs

.. autosignature:: mlprodict.tools.code_helper.make_callable

.. autosignature:: mlprodict.onnx_tools.model_checker.onnx_shaker

.. autosignature:: mlprodict.onnx_tools.optimisation._main_onnx_optim.onnx_optimisations

.. autosignature:: mlprodict.onnx_tools.optim.onnx_statistics

.. autosignature:: mlprodict.onnx_tools.onnx_manipulations.select_model_inputs_outputs

.. autosignature:: mlprodict.testing.verify_code.verify_code

.. autosignature:: mlprodict.testing.script_testing.verify_script

Optimisation
++++++++++++

The following functions reduce the number of ONNX operators in a graph
while keeping the same results. The optimized graph
is left unchanged.

.. autosignature:: mlprodict.onnx_tools.onnx_tools.ensure_topological_order

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation.onnx_remove_node

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation_identity.onnx_remove_node_identity

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation_redundant.onnx_remove_node_redundant

.. autosignature:: mlprodict.onnx_tools.optim.onnx_remove_unused.onnx_remove_node_unused

Profiling
+++++++++

.. autosignature:: mlprodict.tools.ort_wrapper.prepare_c_profiling

Serialization
+++++++++++++

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.from_bytes

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.to_bytes

Runtime
=======

.. autosignature:: mlprodict.tools.onnx_micro_runtime.OnnxMicroRuntime

Validation
++++++++++

.. autosignature:: mlprodict.onnx_tools.model_checker.onnx_shaker

Visualization
+++++++++++++

Many times I had to debug and I was thinking about a way to see
a graph in a text editor. That's the goal of this function with
the possibility later to only show a part of a graph.

.. autosignature:: mlprodict.onnx_tools.graphs.onnx2bigraph

Others
======

Benchmark
+++++++++

.. autosignature:: mlprodict.tools.speed_measure.measure_time

Plotting
++++++++

.. autosignature:: mlprodict.plotting.plotting_benchmark.plot_benchmark_metrics

.. autosignature:: mlprodict.onnxrt.doc.nb_helper.onnxview

.. autosignature:: mlprodict.plotting.plotting_validate_graph.plot_validate_benchmark

Versions
++++++++

.. autosignature:: mlprodict.tools.asv_options_helper.get_ir_version_from_onnx

.. autosignature:: mlprodict.tools.asv_options_helper.get_opset_number_from_onnx
