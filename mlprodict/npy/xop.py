# pylint: disable=E1101
"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import numpy
from scipy.sparse import coo_matrix
from onnx import GraphProto, TensorProto
from onnx.helper import make_graph, make_model  # pylint: disable=W0611
from onnx.numpy_helper import from_array
from .xop_variable import Variable, is_numpy_dtype
from .xop_graph_builder import GraphBuilder


class OnnxOperatorItem:
    """
    Accessor to one of the output returned by a @see cl OnnxOperator.

    :param onx_op: @see cl OnnxOperator
    :param index: integer
    :param op_version: defines the opset version
    """

    def __init__(self, onx_op, index, op_version=None):
        if not isinstance(index, int):
            raise TypeError("index must be an integer not %r." % type(index))
        self.onx_op = onx_op
        self.index = index
        self.op_version = op_version

    def __str__(self):
        """
        usual
        """
        return "%s[%d]" % (str(self.onx_op), self.index)

    def get_output_name(self, i=0):
        """
        Returns the output name at position *i*.
        """
        if i != 0:
            raise IndexError("Can only return the first item.")
        return self.onx_op.get_output_name(self.index)

    def get_output(self, i=0):
        """
        Returns the output.
        """
        if i != 0:
            raise IndexError("Can only return the first item.")
        return self.onx_op.get_output(self.index)


class OnnxOperator:
    """
    Ancestor to every *ONNX* operator exposed in
    :mod:`mlprodict.npy.xops` and :mod:`mlprodict.npy.xops_ml`.

    :param inputs: list of inputs expected by the operator
    :param op_version: to select a specific version of the operator
    :param output_names: used defined names for the outputs
    :param domain: to overwrite the default domain
    :param global_context: operator *If* executes one subgraph
        whose nodes may use one existing output in the current
        context. If not used in the main graph, these operators
        are not linked to the output and cannot be retrieved.
        *global_context* is a dictionary mapped the subgraph input
        names to these operators.
    :param kwargs: additional parameters of the operator

    .. versionadd:: 0.9
    """

    def __init__(self, *inputs, op_version=None, output_names=None,
                 domain=None, global_context=None, **kwargs):

        if (output_names is None and
                self.__class__.__name__.startswith("OnnxScan")):
            raise NotImplementedError(
                "The class cannot infer the number of variables "
                "for node '{}' yet. output_names must be specified"
                ".".format(self.__class__.__name__))
        if isinstance(output_names, (str, Variable)):
            output_names = [output_names]
            if isinstance(output_names[0], str):
                output_names[0] = Variable(output_names[0])
        elif isinstance(output_names, list):
            if len(output_names) == 0:
                raise ValueError(
                    "output_names cannot be empty (operator %r)."
                    "" % self.__class__.__name__)
            output_names = output_names.copy()
            for i in range(len(output_names)):  # pylint: disable=C0200
                if isinstance(output_names[i], str):
                    output_names[i] = Variable(output_names[i])
        elif output_names is not None:
            raise TypeError(
                "output_names must be a string or a list not %r."
                "" % type(output_names))

        if op_version is None:
            if domain == '':
                self.op_version = get_latest_tested_opset_version()
            else:
                self.op_version = None
        else:
            self.op_version = op_version
        self.since_version = self.__class__.since_version

        if (self.op_version is not None and
                self.op_version < self.since_version):
            schema = self.find_schema(self.op_version)
            self.since_version = schema.since_version
            self.expected_inputs = schema.expected_inputs.copy()
            self.expected_outputs = schema.expected_outputs.copy()
            self.input_range = schema.input_range
            self.output_range = schema.output_range
        else:
            self.expected_inputs = (
                None if self.__class__.expected_inputs is None
                else self.__class__.expected_inputs.copy())
            self.expected_outputs = (
                None if self.__class__.expected_outputs is None
                else self.__class__.expected_outputs.copy())
            self.input_range = self.__class__.input_range
            self.output_range = self.__class__.output_range
            if self.__class__.__name__ not in {
                    'OnnxScan', 'OnnxLoop', 'OnnxIf'}:
                # The minimum opset depends on embedded graph
                # by default, it takes the given op_version but the
                # optimal value could be lower.
                self.op_version = self.since_version
            if self.op_version is None:
                self.op_version = self.since_version

        if (self.op_version is not None and
                self.op_version < self.since_version):
            raise RuntimeError(
                "Operator '{}': requested version {} < "
                "{} schema version.".format(
                    self.__class__.__name__,
                    self.op_version, self.since_version))

        self.state = None
        self.domain = domain
        self.kwargs = kwargs
        self.onnx_prefix_name = None

        # check inputs
        if len(inputs) == 0:
            if self.input_range[0] == self.input_range[1]:
                self.inputs = [OnnxOperator.UnscopedVariable(_[0])
                               for _ in self.expected_inputs]
            else:
                # The number of inputs may vary.
                self.inputs = None
        else:
            self.inputs = []
            for inp in inputs:
                if isinstance(inp, str):
                    self.inputs.append(Variable(inp))
                elif isinstance(inp, (OnnxOperator, Variable,
                                      OnnxOperatorItem)):
                    self.inputs.append(inp)
                elif isinstance(inp, (numpy.ndarray, coo_matrix, TensorProto)):
                    self.inputs.append(inp)
                else:
                    raise TypeError(
                        "Unable to interpret the input name for type {} in "
                        "operator '{}' (value={}).".format(
                            type(inp), self.__class__.__name__, inp))

        if self.inputs is not None:
            if (len(self.inputs) < self.input_range[0] or
                    len(self.inputs) > self.input_range[1]):
                raise RuntimeError(
                    "Operator '{}' expects a number of inputs "
                    "in [{}, {}] not {} (expected opset={}, "
                    "class opset={})".format(
                        self.operator_name, *self.input_range,
                        len(self.inputs), op_version, self.op_version))
        # global context
        if global_context is None:
            self.global_context = None
        else:
            if not isinstance(global_context, dict):
                raise TypeError(
                    "global_context must be a dictionary not %r."
                    "" % type(global_context))
            for k, v in global_context.items():
                if not isinstance(v, (OnnxOperator, OnnxOperatorItem)):
                    raise TypeError(
                        "Value %r in must be an OnnxOperator or an "
                        "OnnxOperatorItem not %r." % (k, type(v)))
            self.global_context = global_context

        # check output
        self.output_names = output_names
        self.output_variables = None

        if self.output_names is not None:
            if len(self.output_names) == 0:
                raise ValueError(
                    "output_names can be None but cannot be empty for "
                    "operator %r." % self)
            if self.output_variables is None:
                self.output_variables = [None for o in self.output_names]
            for i in range(len(self.output_names)):  # pylint: disable=C0200
                name = self.output_names[i]
                if isinstance(name, Variable):
                    self.output_variables[i] = name
                else:
                    raise TypeError("output_names must be a list of strings "
                                    "and element %r is %r (%r)" % (
                                        i, type(name), name))
            if all(map(lambda x: x is None, self.output_variables)):
                self.output_variables = None

        if (self.output_names is not None and (
                self.expected_outputs is None or
                len(self.output_names) > len(self.expected_outputs))):
            if self.expected_outputs is None:
                self.expected_outputs = []
            for i in range(len(self.expected_outputs),
                           len(self.output_names)):
                self.expected_outputs.append((self.output_names[i], None))

        if (self.expected_inputs is None or
                len(self.inputs) > len(self.expected_inputs)):
            if self.expected_inputs is None:
                self.expected_inputs = []
            for i in range(len(self.expected_inputs),
                           len(self.inputs)):
                inp = self.inputs[i]
                if isinstance(inp, str):
                    inp = (inp, None)
                elif hasattr(inp, 'add_to'):
                    # OnnxOperator
                    existing = set(_[0] for _ in self.expected_inputs)
                    i = 10
                    name = "input%d" % (10 + i)
                    while name in existing:
                        i += 1
                        name = "input%d" % (10 + i)
                    inp = (name, None)
                self.expected_inputs.append(inp)

        self._post_process_attributes()
        self._check()

    def _check(self):
        input_types = (Variable, OnnxOperator, numpy.ndarray)
        for o in self.inputs:
            if not isinstance(o, input_types):
                raise TypeError(
                    "Wrong type for inputs %r." % (
                        self.inputs, ))
        if self.output_names is not None:
            for o in self.output_names:
                if not isinstance(o, Variable):
                    raise TypeError(
                        "Wrong type for output_names %r." % (
                            self.output_names, ))

    def _post_process_attributes(self):
        """
        Walks through attributes and replaces them by ONNX values.
        """
        # Looks into attributes if there is any tuple
        # (GraphProto, OnnxOperator). In that case, the function
        # replaces the tuple by the graph proto and keeps
        # in attributes graph_algebra the OnnxOperator
        # which is the source of it.
        updates = {}
        graph_algebra = {}
        for k, v in self.kwargs.items():
            if isinstance(v, tuple) and isinstance(v[0], GraphProto):
                updates[k] = v[0]
                graph_algebra[k] = v[1]

        if len(graph_algebra) > 0:
            self.kwargs.update(updates)
            self.graph_algebra = graph_algebra

        if self.__class__.__name__ == "OnnxConstantOfShape":
            if "value" in self.kwargs:
                value = self.kwargs['value']
                if isinstance(value, TensorProto):
                    return
                if isinstance(value, numpy.ndarray):
                    if value.shape == (1, ):
                        val = value[0]
                    elif len(value.shape) == 0:
                        val = value
                    else:
                        raise RuntimeError(
                            "Unexpected shape %r for value, it must be "
                            "an array of one element." % value.shape)
                    self.kwargs['value'] = from_array(
                        numpy.array([val], dtype=value.dtype))
                    return
                raise TypeError(
                    "Unexpected type %r for value. It should be an array "
                    "of one element." % type(value))
            return

        if self.__class__.__name__ == "OnnxCast":
            if "to" in self.kwargs:
                value = self.kwargs['to']
                stop
                self.kwargs['to'] = to
            return

    def find_schema(self, op_version):
        """
        Checks if there is an existing schema for a
        specific version.

        :param op_version: requested version
        :return: schema
        """
        if not hasattr(self.__class__, 'past_version'):
            raise RuntimeError("Missing attribute 'past_version', there is "
                               "no other available schema.")
        found = None
        for v in self.past_version.values():
            if v.since_version > op_version:
                continue
            if found is None or v.since_version > found.since_version:
                found = v
        if found is None:
            raise RuntimeError(
                "Operator '{}': requested version {} < "
                "{} schema version.".format(
                    self.__class__.__name__,
                    op_version, self.since_version))
        return found

    def __str__(self):
        """
        usual
        """
        return "{}({} in) -> {}".format(
            self.__class__.__name__,
            len(self.inputs) if self.inputs is not None else 0,
            [str(o) for o in self.output_names]
            if self.output_names is not None else "?")

    def set_onnx_name_prefix(self, onnx_prefix_name):
        """
        Provides a name to define a prefix in the onnx graph
        to avoid to get unreadable node names. The method
        does not overwrite an existing name, it propagates
        the prefix to inputs and stops the propagation
        if the prefix is already defined.
        """
        if self.onnx_prefix_name is None:
            self.onnx_prefix_name = onnx_prefix_name
            for inp in self.inputs:
                if hasattr(inp, 'set_onnx_prefix_name'):
                    inp.set_onnx_name_prefix(onnx_prefix_name)
        return self

    @property
    def onnx_prefix(self):
        "Returns a prefix for results coming out from this node."
        if self.onnx_prefix_name is None:
            name = self.__class__.__name__
            if name.startswith("Onnx"):
                name = name[4:]
            return 'out_' + name[:3].lower()
        return self.onnx_prefix_name

    def __getitem__(self, index):
        """
        Returns an accessor to one of the output
        of this node.
        """
        return OnnxOperatorItem(self, index, self.op_version)

    def _node_to_graph(self, other_outputs=None, inputs=None, outputs=None):
        """
        Builds a graph as a list of nodes to walk through in that order.
        """
        node_outputs = [self]
        if other_outputs is not None:
            node_outputs += other_outputs

        # walk through graphs
        stack = list(node_outputs)
        new_inputs = []
        set_inputs = set()
        memo = []
        while len(stack) > 0:
            memo.extend(stack)
            new_stack = []
            for obj in stack:
                for inp in obj.inputs:
                    if isinstance(inp, OnnxOperator):
                        new_stack.append(inp)
                    elif isinstance(inp, Variable):
                        if inp.name in set_inputs:
                            continue
                        set_inputs.add(inp.name)
                        if inputs is None:
                            new_inputs.append(inp)
                        elif isinstance(inputs, dict):
                            if inp in inputs:
                                new_inputs.append((inp, inputs[inp]))
                            else:
                                raise ValueError(  # pragma: no cover
                                    "Unable to find input %r in %r." % (
                                        inp, inputs))
                        elif is_numpy_dtype(inputs):
                            new_inputs.append(inp.copy_add(inputs))
                        else:
                            raise RuntimeError(  # pragma: no cover
                                "Unable to handle inputs=%r." % inputs)
                    elif isinstance(inp, numpy.ndarray):
                        pass
                    else:
                        raise TypeError(
                            "Unexpected input type %r in node type %r." % (
                                type(inp), type(obj)))
            stack = new_stack

        if len(new_inputs) == 0:
            raise RuntimeError(
                "No detected inputs inputs=%r outputs=%r." % (
                    inputs, outputs))

        # eliminate duplicates
        done = set()
        nodes = []
        for node in reversed(memo):
            if id(node) in done:
                continue
            done.add(id(node))
            nodes.append(node)

        def _get_type(node, name=None, outputs=None):
            if outputs is None:
                raise NotImplementedError(
                    "outputs is None, expected_outputs=%r" % (
                        node.expected_outputs, ))
            if isinstance(outputs, dict):
                if name is None:
                    raise RuntimeError(
                        "Unable to get type among %r, name=None." % (
                            outputs, ))
                if name not in outputs:
                    raise ValueError(  # pragma: no cover
                        "Unable to find %r in %r." % (
                            name, outputs))
                return outputs[name]
            if isinstance(outputs, list):
                raise NotImplementedError(
                    "Unexpected type for name=%r, outputs=%r." % (
                        name, outputs))
            if is_numpy_dtype(outputs):
                return outputs
            raise RuntimeError(  # pragma: no cover
                "Unable to handle outputs=%r." % outputs)

        # outputs
        new_outputs = []
        for node in node_outputs:
            if node.output_names is None:
                n = self.output_range[0]
                for i in range(n):
                    to = _get_type(node, outputs=outputs)
                    res = ('out%d' % i, to)
                    new_outputs.append(Variable(res[0], added_dtype=to))
            else:
                for o in self.output_names:
                    to = _get_type(node, o, outputs=outputs)
                    res = (o, to)
                    new_outputs.append(o.copy_add(to))
        if len(new_outputs) == 0:
            raise RuntimeError(
                "No detected outputs inputs=%r outputs=%r." % (
                    inputs, outputs))

        return nodes, new_inputs, new_outputs

    def add_to(self, builder):
        """
        Adds to graph builder.
        """
        inputs = builder.get_input_names(self, self.inputs)
        n_outputs = (
            self.output_range[0] if self.output_names is None
            else len(self.output_names))
        outputs = [builder.get_output_name(self, i) for i in range(n_outputs)]
        builder.add_node(
            self.operator_name,
            builder.get_unique_name('_' + self.operator_name.lower()),
            inputs, outputs, domain=self.domain, opset=self.op_version,
            **self.kwargs)

    def to_onnx(self, inputs=None, outputs=None,
                other_outputs=None, target_opset=None,
                verbose=0):
        """
        Converts this operator into an ONNX graph.

        :param inputs: information about type
        :param outputs: information about types
        :param other_outputs: additional nodes to consider
            as graph outputs but not outputs of this particular
            node
        :param target_opset: dictionary with target opset per domain,
            None for the default one
        :param verbose: prints information
        """
        # opsets
        if isinstance(target_opset, dict):
            dom = self.domain or ''
            target_opset = target_opset.get(dom, None)
        elif isinstance(target_opset, int):
            if self.domain not in ('', None):
                # The target_opset is for the domain '' we ignore it.
                target_opset = None
        elif target_opset is not None:
            raise TypeError(
                "target_opset must be a dictionary {domain: "
                "target_opset} not %r for operator %r." % (
                    target_opset, self.__class__.__name__))

        if self.domain in ('', None) and target_opset == 1:
            raise RuntimeError("target_opset cannot be 1.")
        if (self.op_version is not None and target_opset is not None and
                self.op_version > target_opset):
            raise RuntimeError(
                "target_opset={} is lower than the version={} requested "
                "for this node '{}'.".format(
                    target_opset, self.op_version, self.__class__.__name__))

        # inputs, outputs
        if isinstance(inputs, list):
            raise NotImplementedError(
                "Unable to process inputs=%r." % (inputs, ))
        if isinstance(outputs, list):
            raise NotImplementedError(
                "Unable to process outputs=%r." % (outputs, ))

        # get the graph
        nodes, graph_inputs, graph_outputs = self._node_to_graph(
            other_outputs, inputs, outputs)
        if len(nodes) == 0:
            raise RuntimeError(  # pragma: no cover
                "Node list is empty.")
        if verbose > 1:
            for i, n in enumerate(nodes):
                print("nodes[%d]=%r" % (i, n))
            for i, n in enumerate(graph_inputs):
                print("graph_inputs[%d]=%r" % (i, n))
        builder = GraphBuilder()
        for node in nodes:
            node.add_to(builder)

        return builder.to_onnx(inputs=graph_inputs,
                               outputs=graph_outputs,
                               target_opset=target_opset,
                               verbose=verbose)
