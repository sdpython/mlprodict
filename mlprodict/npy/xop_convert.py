"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import numpy
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
        for inp, name in zip(self.model.graph.input, inputs):
            new_name = builder.get_unique_name(inp.name)
            mapped_names[inp.name] = new_name
            builder.add_node(
                'Identity', builder.get_unique_name('_sub_' + name),
                [name], [new_name])

        # adding nodes
        for node in self.model.graph.node:
            new_inputs = []
            for i in node.input:
                if i not in mapped_names:
                    raise RuntimeError(
                        "Unable to find input %r in %r." % (i, mapped_names))
                new_inputs.append(mapped_names[i])
            new_outputs = []
            for o in node.output:
                new_name = builder.get_unique_name(o)
                mapped_names[o] = new_name
                new_outputs.append(new_name)

            atts = {}
            for att in node.attribute:
                if att.type == 2:  # .i
                    value = att.i
                    atts[att.name] = value
                    continue
                if att.type == 6:  # .floats
                    value = list(att.floats)
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
    :param initial_types: the implementation may be wrong in guessing
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
                 initial_types=None, **kwargs):
        if model is None:
            raise ValueError("Model cannot be None.")
        onx = OnnxSubEstimator._to_onnx(
            model, inputs, op_version=op_version, options=options,
            initial_types=initial_types, **kwargs)
        OnnxSubOnnx.__init__(
            self, onx, *inputs, output_names=output_names)
        self.ml_model = model
        self.options = options
        self.initial_types = initial_types
        self.op_version = op_version

    def __repr__(self):
        "usual"
        atts = {}
        for att in ['op_version', 'output_names', 'options',
                    'initial_types']:
            value = getattr(self, att, None)
            if value is not None:
                atts[att] = value
        atts.update(self.kwargs)
        msg = ", ".join("%s=%r" % (k, v) for k, v in atts.items())
        if len(atts) > 0:
            msg = ", " + msg
        return "%s(%r%s)" % (
            self.__class__.__name__, self.ml_model, msg)

    @staticmethod
    def _to_onnx(model, inputs, op_version=None, options=None,
                 initial_types=None, **kwargs):
        """
        Converts a model into ONNX and inserts it into an ONNX graph.

        :param model: a trained machine learned model
        :param inputs: inputs
        :param op_version: opset versions or None to use the latest one
        :param options: options to change the behaviour of the converter
        :param kwargs: additional parameters such as black listed or while listed
            operators
        :return: ONNX model

        The method currently supports models trained with
        :epkg:`scikit-learn`, :epkg:`xgboost`, :epkg`:lightgbm`.
        """
        from sklearn.base import BaseEstimator

        if isinstance(model, BaseEstimator):
            return OnnxSubEstimator._to_onnx_sklearn(
                model, inputs, op_version=op_version, options=options,
                initial_types=initial_types, **kwargs)
        raise RuntimeError(
            "Unable to convert into ONNX model type %r." % type(model))

    @staticmethod
    def _to_onnx_sklearn(model, inputs, op_version=None, options=None,
                         initial_types=None, **kwargs):
        """
        Converts a :epkg:`scikit-learn` model into ONNX
        and inserts it into an ONNX graph. The library relies on
        function @see fn to_onnx and library :epkg:`skearn-onnx`.

        :param model: a trained machine learned model
        :param inputs: inputs
        :param op_version: opset versions or None to use the latest one
        :param initial_types: if None, the input types are guessed from the
            inputs. The function converts into ONNX the previous
            node of the graph and tries to infer the initial_types
            with the little informations it has. It may not work.
            It is recommended to specify this parameter.
        :param options: options to change the behaviour of the converter
        :param kwargs: additional parameters such as black listed or while listed
            operators
        :return: ONNX model

        Default options is `{'zipmap': False}` for a classifier.
        """
        from ..onnx_conv.convert import to_onnx
        if options is None:
            from sklearn.base import ClassifierMixin
            if isinstance(model, ClassifierMixin):
                options = {'zipmap': False}
        if initial_types is None:
            # Let's to infer them from previous nodes.
            raise NotImplementedError(
                "initial_types is None and the method cannot guess the "
                "initial_types of the model.")

        if isinstance(initial_types, numpy.ndarray):
            if len(inputs) != 1:
                raise RuntimeError(
                    "The model has %s inputs but only %d input are "
                    "described in 'initial_types'." % (
                        len(inputs), 1))
            X = initial_types
            initial_types = None
        elif len(inputs) != len(initial_types):
            raise RuntimeError(
                "The model has %s inputs but only %d input are "
                "described in 'initial_types'." % (
                    len(inputs), len(initial_types)))
        else:
            X = None

        onx = to_onnx(model, X, initial_types=initial_types, options=options,
                      rewrite_ops=True, target_opset=op_version, **kwargs)
        return onx