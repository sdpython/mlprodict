"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
from .xop import OnnxOperator


class OnnxSubOnnx(OnnxOperator):
    """
    This operator is used to insert existing ONNX into
    the ONNX graph being built.
    """

    domain = 'mlprodict'
    since_version = 1
    expected_inputs = None
    expected_outputs = None
    input_range = [1, 1e9]
    output_range = [1, 1e9]

    def __init__(self, model, *inputs, output_names=None):
        if model is None:
            raise ValueError("Model cannot be None.")
        if len(inputs) > len(model.graph.input):
            raise RuntimeError(
                "Unexpected number of inputs %r > expected %r." % (
                    len(inputs), len(model.graph.input)))
        if (output_names is not None and
                len(output_names) != len(model.graph.output)):
            raise RuntimeError(
                "Unexpected number of outputs %r != expected %r." % (
                    len(output_names), len(model.graph.output)))
        OnnxOperator.__init__(self, *inputs, output_names=output_names)
        self.model = model

    def __repr__(self):
        "usual"
        atts = {}
        for att in ['output_names']:
            value = getattr(self, att, None)
            if value is not None:
                atts[att] = value
        atts.update(self.kwargs)
        msg = ", ".join("%s=%r" % (k, v) for k, v in atts.items())
        if len(atts) > 0:
            msg = ", " + msg
        return "%s(...%s)" % (
            self.__class__.__name__, msg)

    def add_to(self, builder):
        """
        Adds to graph builder.

        :param builder: instance of @see cl _GraphBuilder,
            it must have a method `add_node`
        """
        inputs = builder.get_input_names(self, self.inputs)
        n_outputs = len(self.model.graph.output)
        outputs = [builder.get_output_name(self, i) for i in range(n_outputs)]

        mapped_names = {}

        # adding initializers
        for init in self.model.graph.initializer:
            new_name = builder.get_unique_name(init.name)
            mapped_names[init.name] = new_name
            builder.add_initializer(new_name, init)

        # linking inputs
        for name in inputs:
            new_name = builder.get_unique_name(name)
            mapped_names[name] = new_name
            builder.add_node(
                'Identity', builder.get_unique_name('_sub_' + name),
                [name], [new_name])

        # adding nodes
        for node in self.model.graph.node:
            new_inputs = [mapped_names[i] for i in node.input]
            new_outputs = []
            for o in node.output:
                new_name = builder.get_unique_name(o)
                mapped_names[o] = new_name
                new_outputs.append(new_name)

            atts = {}
            for att in node.attribute:
                if att.type == 2:
                    value = att.i
                    atts[att.name] = value
                    continue
                raise NotImplementedError(
                    "Unable to copy attribute type %r (%r)." % (
                        att.type, att))

            builder.add_node(
                node.op_type,
                builder.get_unique_name('_sub_' + node.name),
                new_inputs, new_outputs, domain=node.domain, **atts)

        # linking outputs
        for out, name in zip(self.model.graph.output, outputs):
            builder.add_node(
                'Identity', builder.get_unique_name('_sub_' + out.name),
                [mapped_names[out.name]], [name])


class OnnxSubEstimator(OnnxSubOnnx):
    """
    This operator is used to call the converter of a model
    to insert the node coming from the conversion into a
    bigger ONNX graph. It supports model from :epkg:`scikit-learn`
    using :epkg:`sklearn-onnx`.

    :param model: model to convert
    :param inputs: inputs
    :param op_version: targetted opset
    :param options: to rewrite the options used to convert the model
    :param input_types: the implementation may be wrong in guessing
        the input types of the model, this parameter can be used
        to overwrite them, usually a dictionary
        `{ input_name: numpy array as an example }`
    :param kwargs: any other parameters such as black listed or
        white listed operators
    """

    since_version = 1
    expected_inputs = None
    expected_outputs = None
    input_range = [1, 1e9]
    output_range = [1, 1e9]

    def __init__(self, model, *inputs, op_version=None,
                 output_names=None, options=None,
                 input_types=None, **kwargs):
        if model is None:
            raise ValueError("Model cannot be None.")
        OnnxSubOnnx.__init__(
            self, *inputs, op_version=op_version,
            output_names=output_names, **kwargs)
        self.model = model
        self.options = options
        self.input_types = input_types

    def __repr__(self):
        "usual"
        atts = {}
        for att in ['op_version', 'output_names', 'options',
                    'input_types']:
            value = getattr(self, att, None)
            if value is not None:
                atts[att] = value
        atts.update(self.kwargs)
        msg = ", ".join("%s=%r" for k, v in atts.items())
        if len(atts) > 0:
            msg += ", "
        return "%s(%r%s)" % (
            self.__class__.__name__, self.model, msg)

    def add_to(self, builder):
        """
        Adds to graph builder.

        :param builder: instance of @see cl _GraphBuilder,
            it must have a method `add_node`
        """
        raise NotImplementedError()
