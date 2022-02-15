"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import numpy
from onnx.helper import (
    make_node, make_graph, make_model,
    make_tensor_value_info)
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from ..tools.asv_options_helper import get_opset_number_from_onnx


def _default_OPSET_TO_IR_VERSION():
    return {
        1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3,
        7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 7,
        13: 7, 14: 7, 15: 8
    }


class Variable:
    """
    An input to an ONNX graph.
    """

    def __init__(self, name, dtype=None):
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        "usual"
        return "%s(%r, %r)" % (
            self.__class__.__name__, self.name, self.dtype)


class GraphBuilder:
    """
    Graph builder.
    """

    def __init__(self):
        from .xop import OnnxOperator, OnnxOperatorItem
        self.initializer = []
        self.node = []
        self.input = []
        self.output = []
        self.opsets = {}
        self.names = set()
        self.input_names = set()
        self.output_names = {}
        self.cl_onnx_op = OnnxOperator
        self.cl_onnx_op_item = OnnxOperatorItem

    def get_unique_name(self, name):
        """
        Returns a unique name to name an output.
        """
        if not isinstance(name, str):
            raise TypeError(  # pragma: no cover
                "name must be a string not %r." % type(name))
        if name not in self.names:
            self.names.add(name)
            return name
        i = 1
        new_name = "%s_%d" % (name, i)
        while new_name in self.names:
            i += 1
            new_name = "%s_%d" % (name, i)
        self.names.add(new_name)
        return new_name

    def get_output_names(self, node, outputs):
        """
        Returns a new output name for a node if it exists
        or create a new one.
        """
        names = []
        for index, name in enumerate(outputs):
            key = id(node), index
            if key in self.output_names:
                name = self.output_names[key]
            else:
                output = node.output_names[index]
                if isinstance(output, str):
                    n = output
                elif isinstance(output, Variable):
                    n = output.name
                else:
                    raise TypeError(  # pragma: no cover
                        "Unexpected type %r for output %d." % (
                            type(output), index))
                name = self.get_unique_name(n)
                self.output_names[key] = name
            names.append(name)
        return names

    def get_input_names(self, node, inputs):
        """
        Returns input names for node *node* and inputs *inputs*.

        :param node: node
        :param inputs: inputs
        :return: name
        """
        names = []
        for i in inputs:
            if isinstance(i, str):
                names.append(i)
                self.input_names.add(i)
                self.names.add(i)
            elif isinstance(i, Variable):
                names.append(i.name)
                self.names.add(i.name)
                self.input_names.add(i.name)
            elif isinstance(i, self.cl_onnx_op):
                name = self.get_output_name(i, 0)
                names.append(name)
                self.names.add(name)
            elif isinstance(i, self.cl_onnx_op_item):
                name = self.get_output_name(i.onnx_op, i.index)
                names.append(name)
                self.names.add(name)
            else:
                raise TypeError(  # pragma: no cover
                    "Unexpected type for an input %r." % type(i))
        return names

    def add_node(self, op_type, name, inputs, outputs, domain='',
                 opset=None, **attributes):
        """
        Adds a node to the graph.

        :param op_type: operator type
        :param name: node name
        :param inputs: inputs name list
        :param outputs: outputs name list
        :param domain: node domain
        :param opset: node opset
        """
        if not isinstance(inputs, list):
            raise TypeError(  # pragma: no cover
                "inputs must be a list not %r." % type(inputs))
        if not isinstance(outputs, list):
            raise TypeError(  # pragma: no cover
                "inputs must be a list not %r." % type(outputs))
        if any(map(lambda x: not isinstance(x, str), inputs)):
            raise TypeError(  # pragma: no cover
                "inputs must be all strings not %r." % inputs)
        if any(map(lambda x: not isinstance(x, (str, Variable)), outputs)):
            raise TypeError(  # pragma: no cover
                "outputs must be all strings not %r." % outputs)
        if opset is not None:
            if domain not in self.opsets:
                self.opsets[domain] = opset
            else:
                self.opsets[domain] = max(opset, self.opsets[domain])
        node = make_node(op_type, inputs, outputs, name=name,
                         domain=domain)
        self.node.append(node)

    def _process_io(self, inputs, input_names):
        if inputs is None:
            return [
                make_tensor_value_info(
                    'X', TensorProto.FLOAT, None)  # pylint: disable=E1101
                for name in self.input_names]

        if inputs in NP_TYPE_TO_TENSOR_TYPE:
            inputs = [inputs]
        elif numpy.dtype(inputs) in NP_TYPE_TO_TENSOR_TYPE:
            inputs = [inputs]
        if len(input_names) != len(inputs):
            raise RuntimeError(  # pragma: no cover
                "Mismatch between %r and %r." % (input_names, inputs))
        if isinstance(input_names, dict):
            if len(input_names) == 1:
                input_names = list(input_names.values())
            else:
                raise NotImplementedError(
                    "Unexpected %r." % input_names)
        res = []
        for inp, name in zip(inputs, input_names):
            if inp in NP_TYPE_TO_TENSOR_TYPE:
                res.append(
                    make_tensor_value_info(
                        name, NP_TYPE_TO_TENSOR_TYPE[inp], None))
            elif numpy.dtype(inp) in NP_TYPE_TO_TENSOR_TYPE:
                res.append(
                    make_tensor_value_info(
                        name, NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(inp)], None))
            else:
                raise RuntimeError(
                    "Unexpected tuple(%r, %r)." % (inp, name))
        return res

    def to_onnx(self, inputs=None, outputs=None,
                target_opset=None, verbose=0):
        """
        Converts this operator into an ONNX graph.

        :param inputs: specific inputs (as a dictionary) or
            default inputs if not specified
        :param outputs: specific outputs
        :param target_opset: dictionary with target opset per domain,
            None for the default one
        :param verbose: prints information
        :return: onnx graph
        """
        # inputs and outputs
        self.input = self._process_io(inputs, self.input_names)
        self.output = self._process_io(outputs, self.output_names)

        graph = make_graph(
            self.node, 'XOP', self.input, self.output, self.initializer)
        onnx_model = make_model(graph)
        opv = self.opsets.get('', get_opset_number_from_onnx())
        opset2ir = _default_OPSET_TO_IR_VERSION()
        irv = opset2ir.get(opv, max(opset2ir.values()))
        onnx_model.ir_version = irv

        del onnx_model.opset_import[:]  # pylint: disable=E1101
        for k, v in self.opsets.items():
            op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
            op_set.domain = k or ''
            op_set.version = v
        return onnx_model
