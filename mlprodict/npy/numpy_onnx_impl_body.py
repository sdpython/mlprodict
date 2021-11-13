"""
@file
@brief Design to implement graph as parameter.

.. versionadded:: 0.8
"""
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
        self.fct = fct
        self.inputs = inputs
        self.alg_ = None

    def __repr__(self):
        "usual"
        return "%s(...)" % self.__class__.__name__

    def _guess_dtype(self, dtype, from_init=False):
        if not hasattr(self, 'onnx_') or from_init:
            return None
        raise NotImplementedError(
            "Type=%r, dtype=%r." % (type(self), dtype))

    def to_algebra(self, op_version=None):
        """
        Converts the variable into an operator.
        """
        if self.alg_ is not None:
            return self.alg_

        var = self.fct(*self.inputs)
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
        for k, v in self.onnx_op_kwargs.items():
            if not isinstance(v, AttributeGraph):
                continue
            alg = v.to_algebra(op_version=op_version)
            # dtypes = [i._guess_dtype(None) for i in v.inputs]
            onx = alg.to_onnx(target_opset=op_version)
            updates[name] = onx.graph
            removed.append(i)
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
