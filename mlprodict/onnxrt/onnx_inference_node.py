"""
@file
@brief
"""
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
        self.inputs = list(sorted(obj for obj in self.onnx_node.input))
        self.outputs = list(sorted(obj for obj in self.onnx_node.output))

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
        return "'Onnx-{}({}) -> {}".format(
            self.op_type, ", ".join(sorted(self.inputs)),
            ", ".join(sorted(self.outputs)))

    def __repr__(self):
        "usual"
        return self.__str__()

    def setup_runtime(self, runtime=None):
        """
        Loads runtime.

        @param      runtime     runtime options
        """
        if self.desc is None:
            raise AttributeError("desc should not be None.")
        self.ops_ = load_op(self.onnx_node, desc=self.desc,
                            options=runtime)

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
        for name, value in zip(self.outputs, res):
            values[name] = value
