# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ...onnx_tools.onnx2py_helper import guess_dtype
from ._op import OpRun
from ._new_ops import OperatorSchema


class Expression(OpRun):

    atts = {
        'expression': None,
    }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Expression.atts,
                       **options)
        if not hasattr(self.expression, 'run'):
            raise RuntimeError(  # pragma: no cover
                "Parameter 'expression' must have a method 'run', "
                "type {}.".format(type(self.then_branch)))

        self._run_expression = (self.expression.run_in_scan
                                if hasattr(self.expression, 'run_in_scan')
                                else self.expression.run)
        self.additional_inputs = list(self.expression.static_inputs)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "Expression":
            return ExpressionSchema()
        raise RuntimeError(  # pragma: no cover
            "Unable to find a schema for operator '{}'.".format(op_name))

    def need_context(self):
        """
        Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Loop).
        The default answer is `False`.
        """
        return True

    def _run(self, *inputs, named_inputs=None, context=None,  # pylint: disable=W0221
             attributes=None, verbose=0, fLOG=None):

        if verbose > 0 and fLOG is not None:
            fLOG('  -- expression> %r' % list(context))
        outputs = self._run_meth_then(named_inputs, context=context,
                                      attributes=attributes,
                                      verbose=verbose, fLOG=fLOG)
        if verbose > 0 and fLOG is not None:
            fLOG('  -- expression<')
        final = tuple([outputs[name]
                      for name in self.expression.output_names])
        return final

    def _pick_type(self, res, name):
        if name in res:
            return res[name]
        out = {o.name: o for o in self.expression.obj.graph.output}
        if name not in out:
            raise ValueError(
                "Unable to find name=%r in %r or %r." % (
                    name, list(sorted(res)), list(sorted(out))))
        dt = out[name].type.tensor_type.elem_type
        return guess_dtype(dt)


class ExpressionSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl ComplexAbs.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'Expression')
        self.attributes = Expression.atts
