"""
@file
@brief
"""
from .ops import load_op
from onnx.onnx_ml_pb2 import GraphProto  # pylint: disable=C0411


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
            self.op_type, ", ".join(sorted(self.inputs)),
            ", ".join(sorted(self.outputs)))

    def __repr__(self):
        "usual"
        return self.__str__()

    def setup_runtime(self, runtime=None, variables=None, rt_class=None):
        """
        Loads runtime.

        @param      runtime     runtime options
        @param      variables   registered variables created by previous operators
        @param      rt_class    runtime class used to compute
                                prediction of subgraphs
        """
        if self.desc is None:
            raise AttributeError("desc should not be None.")
        self.preprocess_parameters(runtime, rt_class)
        self.ops_ = load_op(self.onnx_node, desc=self.desc,
                            options={'provider': runtime} if runtime else None,
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
            if isinstance(value, GraphProto):
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
            import pprint
            raise RuntimeError("Mismatch number of outputs got {} for names {}.\n{}".format(
                len(res), list(sorted(self.outputs)),
                pprint.pformat(self.desc)))
        for name, value in zip(self.outputs, res):
            values[name] = value
