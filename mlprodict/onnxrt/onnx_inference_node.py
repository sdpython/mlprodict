"""
@file
@brief OnnxInferenceNode definition.
"""
import pprint
import numpy
from onnx import onnx_pb as onnx_proto
from .ops import load_op


class OnnxInferenceNode:
    """
    A node to execute.
    """

    def __init__(self, onnx_node, desc):
        """
        @param      onnx_node       onnx_node
        @param      desc            internal description
        """
        if desc is None:
            raise ValueError("desc should not be None.")
        self.desc = desc
        self.onnx_node = onnx_node
        self._init()

    def _init(self):
        """
        Prepares the node.
        """
        self.op_type = self.onnx_node.op_type
        self.order = -1
        self.variable_to_clean = []
        self.inputs = list(self.onnx_node.input)
        self.outputs = list(self.onnx_node.output)
        self.inplaces = []

    def set_order(self, order):
        """
        Defines the order of execution.
        """
        self.order = order

    def add_variable_to_clean(self, name):
        """
        Adds a variable which can be cleaned after the node
        execution.
        """
        self.variable_to_clean.append(name)

    def __str__(self):
        "usual"
        return "Onnx-{}({}) -> {}".format(
            self.op_type, ", ".join(self.inputs),
            ", ".join(self.outputs))

    def __repr__(self):
        "usual"
        return self.__str__()

    def setup_runtime(self, runtime=None, variables=None, rt_class=None,
                      target_opset=None, dtype=None):
        """
        Loads runtime.

        @param      runtime         runtime options
        @param      variables       registered variables created by previous operators
        @param      rt_class        runtime class used to compute
                                    prediction of subgraphs
        @param      target_opset    use a specific target opset
        @param      dtype           float computational type
        """
        if self.desc is None:
            raise AttributeError("desc should not be None.")
        self.preprocess_parameters(runtime, rt_class)
        options = {'provider': runtime} if runtime else {}
        if target_opset is not None:
            options['target_opset'] = target_opset
        if runtime == 'onnxruntime2':
            self.ops_ = load_op(self.onnx_node, desc=self.desc,
                                options=options if options else None,
                                variables=variables, dtype=dtype)
        else:
            self.ops_ = load_op(self.onnx_node, desc=self.desc,
                                options=options if options else None,
                                variables=variables)

    def preprocess_parameters(self, runtime, rt_class):
        """
        Preprocesses the parameters,
        loads *GraphProto*
        (equivalent to :epkg:`ONNX` graph with
        less metadata).

        @param      runtime     runtime options
        @param      rt_class    runtime class used to compute
                                prediction of subgraphs
        """
        if 'atts' not in self.desc:
            return
        for _, v in self.desc['atts'].items():
            if 'value' not in v:
                continue
            value = v['value']
            if isinstance(value, onnx_proto.GraphProto):
                sess = rt_class(v['value'], runtime=runtime)
                v['value_rt'] = sess

    def run(self, values):
        """
        Runs the node.
        the function updates values with outputs.

        @param      values      list of existing values
        """
        args = [values[k] for k in self.inputs]
        res = self.ops_.run(*args)
        if not isinstance(res, tuple):
            raise RuntimeError("Results of an operator should be a tuple.")
        if len(self.outputs) != len(res):
            raise RuntimeError("Mismatch number of outputs got {} for names {}.\n{}".format(
                len(res), list(sorted(self.outputs)),
                pprint.pformat(self.desc)))
        for name, value in zip(self.outputs, res):
            values[name] = value

    def switch_initializers_dtype(self, dtype_in=numpy.float32,
                                  dtype_out=numpy.float64):
        """
        Switches all initializers to ``numpy.float64``.
        This only works if the runtime is ``'python'``.

        @param      dtype_in    previous type
        @param      dtype_out   next type
        @return                 done operations
        """
        done = []
        for k, v in self.desc['atts'].items():
            if 'value_rt' not in v:
                continue
            if isinstance(v['value_rt'], numpy.ndarray):
                if v['value_rt'].dtype == dtype_in:
                    v['value_rt'] = v['value_rt'].astype(dtype_out)
                    done.append(("+", "desc", k, v['value_rt']))
                else:
                    done.append(("-", "desc", k, v['value_rt']))
        if hasattr(self, 'ops_') and self.ops_ is not None:
            res = self.ops_.switch_initializers_dtype(dtype_in, dtype_out)
            for r in res:
                done.append(("ops_", ) + r)
        return done

    def _set_shape_inference_runtime(self, values):
        """
        Updates *values* which shapes of the outputs.

        @param      values      container for shapes
        """
        args = [values[k] for k in self.inputs]
        res = self.ops_.infer_shapes(*args)
        if not isinstance(res, tuple):
            raise RuntimeError(
                "Results of an operator should be a tuple for operator '{}'"
                ".".format(type(self.ops_)))
        if len(self.outputs) != len(res):
            raise RuntimeError(
                "Mismatch number of outputs got {} for names {} (node='{}')."
                "\n{}".format(
                    len(res), list(self.outputs),
                    self.ops_.__class__.__name__,
                    pprint.pformat(self.desc)))
        for name, value in zip(self.outputs, res):
            values[name] = value

    def enable_inplace_compute(self, name):
        """
        Let the node know that one input can be overwritten.

        @param      name        input name
        """
        self.inplaces.append(name)
        self.ops_.enable_inplace_compute(self.inputs.index(name))
