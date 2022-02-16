"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import numpy
from onnx import TensorProto
from onnx.helper import (
    make_node, make_graph, make_model,
    make_tensor_value_info)
from onnx.numpy_helper import from_array
from ..tools.asv_options_helper import get_opset_number_from_onnx
from .xop_variable import Variable, is_numpy_dtype


def _default_OPSET_TO_IR_VERSION():
    """
    Returns the default mapping between opset and ir_version.

    .. runpython::
        :showcode:

        import pprint
        from mlprodict.npy.xop_graph_builder import _default_OPSET_TO_IR_VERSION
        pprint.pprint(_default_OPSET_TO_IR_VERSION())
    """
    return {
        1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3,
        7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 7,
        13: 7, 14: 7, 15: 8
    }


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
        self.input_names = {}
        self.output_names = {}
        self.output_names_rev = {}
        self.cl_onnx_op = OnnxOperator
        self.cl_onnx_op_item = OnnxOperatorItem

    @staticmethod
    def number2alpha(index):
        """
        Converts a numbers into a string keeping the same
        alphabetical order.
        """
        dec = str(int(index))
        if len(dec) == 1:
            return dec
        return chr(96 + len(dec)) + dec

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
        new_name = "%s_%s" % (name, self.number2alpha(i))
        while new_name in self.names:
            i += 1
            new_name = "%s_%s" % (name, self.number2alpha(i))
        self.names.add(new_name)
        return new_name

    def get_output_name(self, node, index):
        """
        Returns the output name for a node.
        """
        key = id(node), index
        if key in self.output_names:
            name = self.output_names[key]
            return name

        if node.output_names is None:
            prefix = node.onnx_prefix
            n = '%s%d' % (prefix, index)
        else:
            output = node.output_names[index]
            if isinstance(output, Variable):
                n = output.name
            else:
                raise TypeError(  # pragma: no cover
                    "Unexpected type %r for output %d (output_names=%r)." % (
                        type(output), index, node.output_names))

        name = self.get_unique_name(n)
        self.output_names[key] = name
        self.output_names_rev[name] = key
        if node.output_names is not None:
            var = node.output_names[index]
            if isinstance(var, Variable):
                var = var.name
            if var != name:
                raise RuntimeError(
                    "Output unique name %r is different from the "
                    "expected name %r at position %r." % (
                        name, node.output_names[index], index))
        return name

    def get_input_names(self, node, inputs):
        """
        Returns input names for node *node* and inputs *inputs*.

        :param node: node
        :param inputs: inputs
        :return: name
        """
        names = []
        for i in inputs:
            if isinstance(i, Variable):
                names.append(i.name)
                self.names.add(i.name)
                self.input_names[i.name] = i
            elif isinstance(i, self.cl_onnx_op):
                name = self.get_output_name(i, 0)
                names.append(name)
                self.names.add(name)
            elif isinstance(i, self.cl_onnx_op_item):
                name = self.get_output_name(i.onnx_op, i.index)
                names.append(name)
                self.names.add(name)
            elif isinstance(i, numpy.ndarray):
                # Adding an initializer
                name = self.get_unique_name('init')
                init = from_array(i, name)
                self.initializer.append(init)
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
                         domain=domain, **attributes)
        self.node.append(node)

    def _process_io(self, inputs, input_names):
        if inputs is None:
            return [
                make_tensor_value_info(
                    'X', TensorProto.FLOAT, None)  # pylint: disable=E1101
                for name in self.input_names]

        if not isinstance(inputs, list):
            if is_numpy_dtype(inputs):
                inputs = [inputs]

        if input_names is None:
            # outputs
            input_names = []
            for inp in inputs:
                if isinstance(inp, Variable):
                    if inp.name in self.output_names_rev:
                        input_names.append(inp)
                elif isinstance(inp, tuple) and len(inp) == 2:
                    var, dtype = inp
                    if var.name in self.output_names_rev:
                        input_names.append(Variable(var.name, dtype))
                else:
                    raise TypeError(
                        "Unexpected type %r in %r." % (inp, inputs))
            if len(input_names) == 0:
                raise RuntimeError(
                    "Unable to cross %r and %r." % (input, self.output_names_rev))
        elif not isinstance(input_names, list):
            raise RuntimeError(
                "Unexpected type for input_names %r." % type(input_names))

        if len(input_names) != len(inputs):
            raise RuntimeError(  # pragma: no cover
                "Mismatch between %r and %r." % (
                    input_names, inputs))

        res = []
        for inp, var in zip(inputs, input_names):
            if isinstance(inp, (str, tuple)):
                raise TypeError(
                    "inp not Variable but %r (%r)." % (type(inp), inp))
            if isinstance(var, (str, tuple)):
                raise TypeError(
                    "var not Variable but %r (%r)." % (type(var), var))
            if isinstance(var, (str, tuple)):
                raise TypeError(
                    "var not Variable but %r (%r)." % (type(var), var))
            # inp: Variable
            # var: str
            if inp != var:
                raise RuntimeError(
                    "Unexpected %r != %r." % (inp, var))
            res.append(make_tensor_value_info(
                inp.name, inp.proto_added_type, None))

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
        self.input = self._process_io(inputs, list(self.input_names.values()))
        self.output = self._process_io(outputs, None)

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
