"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import logging
import numpy
from .xop import OnnxOperator, OnnxOperatorFunction
from .xop_variable import NodeResultName, Variable


logger = logging.getLogger('xop')


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
    op_type = 'SubOnnx'
    domain = 'mlprodict.xop'

    def __init__(self, model, *inputs, output_names=None):
        logger.debug("SubOnnx(ONNX, %d in, output_names=%r)",
                     len(inputs), output_names)
        if model is None:
            raise ValueError("Model cannot be None.")  # pragma: no cover
        if len(inputs) > len(model.graph.input):
            raise RuntimeError(  # pragma: no cover
                "Unexpected number of inputs %r > expected %r." % (
                    len(inputs), len(model.graph.input)))
        if (output_names is not None and
                len(output_names) != len(model.graph.output)):
            raise RuntimeError(  # pragma: no cover
                "Unexpected number of outputs %r != expected %r." % (
                    len(output_names), len(model.graph.output)))
        if len(inputs) == 0:
            if hasattr(model, 'graph'):
                inputs = [Variable(i.name, i.type.tensor_type)
                          for i in model.graph.input]
            else:
                inputs = [Variable(n) for n in model.input]
        OnnxOperator.__init__(self, *inputs, output_names=output_names)
        if self.output_names is None and self.expected_outputs is None:
            if hasattr(model, 'graph'):
                self.expected_outputs = [
                    (i.name, i.type.tensor_type)
                    for i in model.graph.output]
            else:
                self.expected_outputs = [(n, None) for n in model.output]
        self.model = model

    @property
    def input_names(self):
        "Returns the input names."
        return ([i.name for i in self.model.graph.input]
                if hasattr(self.model, 'graph') else list(self.model.input))

    def __repr__(self):
        "usual"
        atts = {}
        for att in ['output_names']:
            value = getattr(self, att, None)
            if value is not None:
                atts[att] = value
        atts.update(self.kwargs)
        msg = ", ".join(f"{k}={v!r}" for k, v in atts.items())
        if len(atts) > 0:
            msg = ", " + msg
        return f"{self.__class__.__name__}(...{msg})"

    def add_to(self, builder):
        """
        Adds to graph builder.

        :param builder: instance of @see cl _GraphBuilder,
            it must have a method `add_node`
        """
        logger.debug("SubOnnx.add_to(builder)")
        inputs = builder.get_input_names(self, self.inputs)
        n_outputs = len(self.model.graph.output)
        outputs = [builder.get_unique_output_name(NodeResultName(self, i))
                   for i in range(n_outputs)]

        mapped_names = {}

        # adding initializers
        for init in self.model.graph.initializer:
            new_name = builder.get_unique_name(init.name, reserved=False)
            mapped_names[init.name] = new_name
            builder.add_initializer(new_name, init)

        # linking inputs
        for inp, name in zip(self.model.graph.input, inputs):
            new_name = builder.get_unique_name(inp.name, reserved=False)
            mapped_names[inp.name] = new_name
            builder.add_node(
                'Identity', builder.get_unique_name(
                    '_sub_' + name, reserved=False),
                [name], [new_name])

        # adding nodes
        for node in self.model.graph.node:
            new_inputs = []
            for i in node.input:
                if i not in mapped_names:
                    raise RuntimeError(  # pragma: no cover
                        f"Unable to find input {i!r} in {mapped_names!r}.")
                new_inputs.append(mapped_names[i])
            new_outputs = []
            for o in node.output:
                new_name = builder.get_unique_name(o, reserved=False)
                mapped_names[o] = new_name
                new_outputs.append(new_name)

            atts = {}
            for att in node.attribute:
                atts[att.name] = OnnxOperatorFunction.attribute_to_value(att)

            builder.add_node(
                node.op_type,
                builder.get_unique_name('_sub_' + node.name, reserved=False),
                new_inputs, new_outputs, domain=node.domain, **atts)

        # linking outputs
        for out, name in zip(self.model.graph.output, outputs):
            builder.add_node(
                'Identity', builder.get_unique_name(
                    '_sub_' + out.name, reserved=False),
                [mapped_names[out.name]], [name])

    def to_onnx_this(self, evaluated_inputs):
        """
        Returns the ONNX graph.

        :param evaluated_inputs: unused
        :return: ONNX graph
        """
        return self.model


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
    op_type = "SubEstimator"
    domain = 'mlprodict.xop'

    def __init__(self, model, *inputs, op_version=None,
                 output_names=None, options=None,
                 initial_types=None, **kwargs):
        logger.debug("OnnxSubEstimator(%r, %r, op_version=%r, "
                     "output_names=%r, initial_types=%r, options=%r, "
                     "kwargs=%r)", type(model), inputs, op_version,
                     output_names, initial_types, options, kwargs)
        if model is None:
            raise ValueError("Model cannot be None.")  # pragma: no cover
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
        msg = ", ".join(f"{k}={v!r}" for k, v in atts.items())
        if len(atts) > 0:
            msg = ", " + msg
        return f"{self.__class__.__name__}({self.ml_model!r}{msg})"

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
            logger.debug("OnnxSubEstimator._to_onnx(%r, %r, op_version=%r "
                         "options=%r, initial_types=%r, kwargs=%r)",
                         type(model), inputs, op_version, options,
                         initial_types, kwargs)
            return OnnxSubEstimator._to_onnx_sklearn(
                model, inputs, op_version=op_version, options=options,
                initial_types=initial_types, **kwargs)
        raise RuntimeError(  # pragma: no cover
            f"Unable to convert into ONNX model type {type(model)!r}.")

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
            # adding more information
            from skl2onnx.common.data_types import _guess_numpy_type  # delayed
            for i, n in enumerate(inputs):
                if not isinstance(n, Variable):
                    raise NotImplementedError(
                        "Inpput %d is not a variable but %r." % (i, type(n)))
            initial_types = [(n.name, _guess_numpy_type(n.dtype, n.shape))
                             for n in inputs]

        logger.debug("OnnxSubEstimator._to_onnx_sklearn(%r, %r, "
                     "op_version=%r, options=%r, initial_types=%r, "
                     "kwargs=%r)",
                     type(model), inputs, op_version, options,
                     initial_types, kwargs)

        if isinstance(initial_types, numpy.ndarray):
            if len(inputs) != 1:
                raise RuntimeError(  # pragma: no cover
                    "The model has %s inputs but only %d input are "
                    "described in 'initial_types'." % (
                        len(inputs), 1))
            X = initial_types
            initial_types = None
        elif len(inputs) != len(initial_types):
            raise RuntimeError(  # pragma: no cover
                "The model has %s inputs but only %d input are "
                "described in 'initial_types'." % (
                    len(inputs), len(initial_types)))
        else:
            X = None

        onx = to_onnx(model, X, initial_types=initial_types, options=options,
                      rewrite_ops=True, target_opset=op_version, **kwargs)
        return onx
