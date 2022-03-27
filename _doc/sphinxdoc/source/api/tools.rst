
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

Graphs helper, manipulations
++++++++++++++++++++++++++++

Functions to help understand models or modify them.

.. autosignature:: mlprodict.tools.model_info.analyze_model

.. autosignature:: mlprodict.onnx_tools.onnx_manipulations.insert_results_into_onnx

.. autosignature:: mlprodict.onnx_tools.onnx_manipulations.enumerate_model_node_outputs

.. autosignature:: mlprodict.tools.code_helper.make_callable

.. autosignature:: mlprodict.onnx_tools.model_checker.onnx_shaker

.. autosignature:: mlprodict.onnx_tools.optim.onnx_helper.onnx_statistics

.. autosignature:: mlprodict.onnx_tools.onnx_manipulations.select_model_inputs_outputs

.. autosignature:: mlprodict.testing.verify_code.verify_code

.. autosignature:: mlprodict.testing.script_testing.verify_script

Onnx Optimization
+++++++++++++++++

The following functions reduce the number of ONNX operators in a graph
while keeping the same results. The optimized graph
is left unchanged.

.. autosignature:: mlprodict.onnx_tools.onnx_tools.ensure_topological_order

.. autosignature:: mlprodict.onnx_tools.onnx_manipulations.onnx_rename_names

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation.onnx_remove_node

.. autosignature:: mlprodict.onnx_tools.optim._main_onnx_optim.onnx_optimisations

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation_identity.onnx_remove_node_identity

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation_redundant.onnx_remove_node_redundant

.. autosignature:: mlprodict.onnx_tools.optim.onnx_optimisation_unused.onnx_remove_node_unused

Onnx Schemas
++++++++++++

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.get_onnx_schema

Profiling
+++++++++

.. autosignature:: mlprodict.tools.ort_wrapper.prepare_c_profiling

Serialization
+++++++++++++

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.from_bytes

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.to_bytes

Validation of scikit-learn models
+++++++++++++++++++++++++++++++++

.. autosignature:: mlprodict.onnxrt.validate.validate.enumerate_validated_operator_opsets

.. autosignature:: mlprodict.onnx_tools.model_checker.onnx_shaker

.. autosignature:: mlprodict.onnxrt.validate.side_by_side.side_by_side_by_values

.. autosignature:: mlprodict.onnxrt.validate.validate_summary.summary_report

Visualization
+++++++++++++

.. index:: plotting, plot

Many times I had to debug and I was thinking about a way to see
a graph in a text editor. That's the goal of this function with
the possibility later to only show a part of a graph.

**text**

.. autosignature:: mlprodict.plotting.text_plot.onnx_simple_text_plot

.. autosignature:: mlprodict.plotting.text_plot.onnx_text_plot

.. autosignature:: mlprodict.plotting.text_plot.onnx_text_plot_tree

**drawings**

.. autosignature:: mlprodict.plotting.plotting_onnx.plot_onnx

**notebook**

:ref:`onnxview <l-NB2>`, see also :ref:`numpyapionnxftrrst`.

**benchmark**

.. autosignature:: mlprodict.plotting.plot_validate_benchmark

.. autosignature:: mlprodict.plotting.plotting_benchmark.plot_benchmark_metrics

**notebook**

.. autosignature:: mlprodict.nb_helper.onnxview

Others
======

scikit-learn
++++++++++++

.. autosignature:: mlprodict.grammar.grammar_sklearn.g_sklearn_main.sklearn2graph

Versions
++++++++

.. autosignature:: mlprodict.get_ir_version

.. autosignature:: mlprodict.__max_supported_opset__

.. autosignature:: mlprodict.__max_supported_opsets__

Type conversion
===============

.. autosignature:: mlprodict.onnx_conv.convert.guess_initial_types

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.guess_numpy_type_from_string

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.guess_numpy_type_from_dtype

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.guess_proto_dtype

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.guess_proto_dtype_name

.. autosignature:: mlprodict.onnx_tools.onnx2py_helper.guess_dtype

In :epkg:`sklearn-onnx`:

* `skl2onnx.algebra.type_helper.guess_initial_types`
* `skl2onnx.common.data_types.guess_data_type`
* `skl2onnx.common.data_types.guess_numpy_type`
* `skl2onnx.common.data_types.guess_proto_type`
* `skl2onnx.common.data_types.guess_tensor_type`
* `skl2onnx.common.data_types._guess_type_proto`
* `skl2onnx.common.data_types._guess_numpy_type`

The last example summarizes all the possibilities.

.. runpython::
    :showcode:
    :process:

    import numpy
    from onnx import TensorProto

    from skl2onnx.algebra.type_helper import guess_initial_types
    from skl2onnx.common.data_types import guess_data_type
    from skl2onnx.common.data_types import guess_numpy_type
    from skl2onnx.common.data_types import guess_proto_type
    from skl2onnx.common.data_types import guess_tensor_type
    from skl2onnx.common.data_types import _guess_type_proto
    from skl2onnx.common.data_types import _guess_numpy_type
    from skl2onnx.common.data_types import DoubleTensorType

    from mlprodict.onnx_conv.convert import guess_initial_types as guess_initial_types_mlprodict
    from mlprodict.onnx_tools.onnx2py_helper import guess_numpy_type_from_string
    from mlprodict.onnx_tools.onnx2py_helper import guess_numpy_type_from_dtype
    from mlprodict.onnx_tools.onnx2py_helper import guess_proto_dtype
    from mlprodict.onnx_tools.onnx2py_helper import guess_proto_dtype_name
    from mlprodict.onnx_tools.onnx2py_helper import guess_dtype

    def guess_initial_types0(t):
        return guess_initial_types(numpy.array([[0, 1]], dtype=t), None)

    def guess_initial_types1(t):
        return guess_initial_types(None, [('X', t)])

    def guess_initial_types_mlprodict0(t):
        return guess_initial_types_mlprodict(numpy.array([[0, 1]], dtype=t), None)

    def guess_initial_types_mlprodict1(t):
        return guess_initial_types_mlprodict(None, [('X', t)])

    def _guess_type_proto1(t):
        return _guess_type_proto(t, [None, 4])

    def _guess_numpy_type1(t):
        return _guess_numpy_type(t, [None, 4])

    fcts = [guess_initial_types0, guess_initial_types1,
            guess_data_type, guess_numpy_type,
            guess_proto_type, guess_tensor_type,
            _guess_type_proto1,
            _guess_numpy_type1,
            guess_initial_types_mlprodict0,
            guess_initial_types_mlprodict1,
            guess_numpy_type_from_string,
            guess_numpy_type_from_dtype,
            guess_proto_dtype_name, guess_dtype]

    values = [numpy.float64, float, 'double', 'tensor(double)',
              DoubleTensorType([None, 4]),
              TensorProto.DOUBLE]

    print("---SUCCESS------------")
    errors = []
    for f in fcts:
        print("")
        for v in values:
            try:
                r = f(v)
                print("%s(%r) -> %r" % (f.__name__, v, r))
            except Exception as e:
                errors.append("%s(%r) -> %r" % (f.__name__, v, e))
        errors.append("")

    print()
    print('---ERRORS-------------')
    print()
    for e in errors:
        print(e)

skl2onnx
========

.. autosignature:: mlprodict.onnx_tools.exports.skl2onnx_helper.add_onnx_graph
