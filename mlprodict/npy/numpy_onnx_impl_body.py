"""
@file
@brief Design to implement graph as parameter.

.. versionadded:: 0.8
"""
import numpy
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity)
from .onnx_variable import OnnxVar


class AttributeGraph:
    """
    Class wrapping a function to make it simple as
    a parameter.

    :param fct: function taking the list of inputs defined
        as @see cl OnnxVar, the function returns an @see cl OnnxVar
    :param inputs: list of input as @see cl OnnxVar

    .. versionadded:: 0.8
    """

    def __init__(self, fct, *inputs):
        if isinstance(fct, numpy.ndarray) and len(inputs) == 0:
            self.cst = fct
            fct = None
        else:
            self.cst = None
        self.fct = fct
        self.inputs = inputs
        self.alg_ = None

    def __repr__(self):
        "usual"
        return "%s(...)" % self.__class__.__name__

    def _graph_guess_dtype(self, i, var):
        """
        Guesses the graph inputs.

        :param i: attribute index (integer)
        :param var: the input (@see cl OnnxVar)
        :return: input type
        """
        dtype = var._guess_dtype(None)
        if dtype is None:
            dtype = numpy.float32

        if dtype == numpy.float32:
            skl2onnx_type = FloatTensorType()
        else:
            raise TypeError(
                "Unexpected type %r." % dtype)

        input_type = ('graph_%d_%d' % (id(self), i),
                      skl2onnx_type)
        var.set_onnx_name(input_type)
        return input_type, OnnxVar(input_type[0], dtype=dtype)

    def to_algebra(self, op_version=None):
        """
        Converts the variable into an operator.
        """
        if self.alg_ is not None:
            return self.alg_

        if self.cst is not None:
            self.alg_ = OnnxIdentity(self.cst, op_version=op_version)
            self.alg_inputs_ = None
            return self.alg_

        new_inputs = [self._graph_guess_dtype(i, inp)
                      for i, inp in enumerate(self.inputs)]
        self.alg_inputs_ = new_inputs
        vars = [v[1] for v in new_inputs]
        var = self.fct(*vars)
        if not isinstance(var, OnnxVar):
            raise RuntimeError(  # pragma: no cover
                "var is not from type OnnxVar but %r." % type(var))

        self.alg_ = var.to_algebra(op_version=op_version)
        return self.alg_


class OnnxVarGraph(OnnxVar):
    """
    Overloads @see cl OnnxVar to handle graph attribute.

    :param inputs: variable name or object
    :param op: :epkg:`ONNX` operator
    :param select_output: if multiple output are returned by
        ONNX operator *op*, it takes only one specifed by this
        argument
    :param dtype: specifies the type of the variable
        held by this class (*op* is None) in that case
    :param fields: list of attributes with the graph type
    :param kwargs: addition argument to give operator *op*

    .. versionadded:: 0.8
    """

    def __init__(self, *inputs, op=None, select_output=None,
                 dtype=None, **kwargs):
        OnnxVar.__init__(
            self, *inputs, op=op, select_output=select_output,
            dtype=dtype, **kwargs)

    def to_algebra(self, op_version=None):
        """
        Converts the variable into an operator.
        """
        if self.alg_ is not None:
            return self.alg_

        # Conversion of graph attributes from InputGraph
        # ONNX graph.
        updates = dict()
        self.alg_hidden_var_ = {}
        self.alg_hidden_var_inputs = {}
        for att, var in self.onnx_op_kwargs.items():
            if not isinstance(var, AttributeGraph):
                continue
            alg = var.to_algebra(op_version=op_version)
            alg.set_onnx_name_prefix("g_%s_%d" % (att, id(var)))
            if var.alg_inputs_ is None:
                onnx_inputs = []
            else:
                onnx_inputs = [i[0] for i in var.alg_inputs_]
            onx = alg.to_onnx(onnx_inputs, target_opset=op_version)
            updates[att] = onx.graph
            self.alg_hidden_var_[id(var)] = var
            self.alg_hidden_var_inputs[id(var)] = onnx_inputs
        self.onnx_op_kwargs_before = {
            k: self.onnx_op_kwargs[k] for k in updates}
        self.onnx_op_kwargs.update(updates)
        return OnnxVar.to_algebra(self, op_version=op_version)


class if_then_else(AttributeGraph):
    """
    Overloads class @see cl OnnxVarGraph.
    """

    def __init__(self, fct, *inputs):
        AttributeGraph.__init__(self, fct, *inputs)
