# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ...onnx_tools.onnx2py_helper import guess_dtype
from ..shape_object import ShapeObject
from ._op import OpRun


class If(OpRun):

    atts = {
        'then_branch': None,
        'else_branch': None,
    }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=If.atts,
                       **options)
        if not hasattr(self.then_branch, 'run'):
            raise RuntimeError(  # pragma: no cover
                "Parameter 'then_branch' must have a method 'run', "
                "type {}.".format(type(self.then_branch)))
        if not hasattr(self.else_branch, 'run'):
            raise RuntimeError(  # pragma: no cover
                "Parameter 'else_branch' must have a method 'run', "
                "type {}.".format(type(self.else_branch)))

        self._run_meth_then = (self.then_branch.run_in_scan
                               if hasattr(self.then_branch, 'run_in_scan')
                               else self.then_branch.run)
        self._run_meth_else = (self.else_branch.run_in_scan
                               if hasattr(self.else_branch, 'run_in_scan')
                               else self.else_branch.run)
        self.additional_inputs = list(
            set(self.then_branch.static_inputs) |
            set(self.else_branch.static_inputs))

    def need_context(self):
        """
        Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Loop).
        The default answer is `False`.
        """
        return True

    def _run(self, cond, named_inputs=None, context=None):  # pylint: disable=W0221
        if cond is None:
            raise RuntimeError(  # pragma: no cover
                "cond cannot be None")
        if named_inputs is None:
            named_inputs = {}
        if len(self.then_branch.input_names) > 0:
            if len(named_inputs) == 0:
                raise RuntimeError(  # pragma: no cover
                    "named_inputs is empty but the graph needs {}.".format(
                        self.then_branch.input_names))
            for k in self.then_branch.input_names:
                if k not in named_inputs:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to find named input '{}' in\n{}.".format(
                            k, "\n".join(sorted(named_inputs))))
        if len(self.else_branch.input_names) > 0:
            if len(named_inputs) == 0:
                raise RuntimeError(  # pragma: no cover
                    "named_inputs is empty but the graph needs {}.".format(
                        self.then_branch.input_names))
            for k in self.else_branch.input_names:
                if k not in named_inputs:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to find named input '{}' in\n{}.".format(
                            k, "\n".join(sorted(named_inputs))))

        if len(cond.shape) > 0:
            if all(cond):
                outputs = self._run_meth_then(named_inputs, context=context)
                final = tuple([outputs[name] for name in self.then_branch.output_names])
                branch = 'then'
            else:
                outputs = self._run_meth_else(named_inputs, context=context)
                final = tuple([outputs[name] for name in self.else_branch.output_names])
                branch = 'else'
        elif cond:
            outputs = self._run_meth_then(named_inputs, context=context)
            final = tuple([outputs[name] for name in self.then_branch.output_names])
            branch = 'then'
        else:
            outputs = self._run_meth_else(named_inputs, context=context)
            final = tuple([outputs[name] for name in self.else_branch.output_names])
            branch = 'else'

        if len(final) == 0:
            raise RuntimeError(  # pragma: no cover
                "Operator If (%r) does not have any output." % (self.onnx_node.name, ))
        for i, f in enumerate(final):
            if f is None:
                ni = named_inputs if named_inputs else []  # pragma: no cover
                br = self.then_branch if branch == 'then' else self.else_branch
                names = br.output_names
                inits = [i.name for i in br.obj.graph.initializer]
                raise RuntimeError(  # pragma: no cover
                    "Output %d (branch=%r, name=%r) is None, available inputs=%r, "
                    "initializers=%r." % (
                        i, branch, names[i], list(sorted(ni)), inits))
        return final

    def _pick_shape(self, res, name):
        if name in res and res[name] is not None:
            return res[name]
        out = {o.name: o for o in self.then_branch.obj.graph.output}
        if name not in out:
            raise ValueError(  # pragma: no cover
                "Unable to find name=%r in %r or %r." % (
                    name, list(sorted(res)), list(sorted(out))))
        dt = out[name].type.tensor_type.elem_type
        return ShapeObject(None, guess_dtype(dt))

    def _infer_shapes(self, cond, named_inputs=None):  # pylint: disable=W0221
        res = self.then_branch._set_shape_inference_runtime()
        return tuple([self._pick_shape(res, name)
                     for name in self.then_branch.output_names])

    def _pick_type(self, res, name):
        if name in res:
            return res[name]
        out = {o.name: o for o in self.then_branch.obj.graph.output}
        if name not in out:
            raise ValueError(
                "Unable to find name=%r in %r or %r." % (
                    name, list(sorted(res)), list(sorted(out))))
        dt = out[name].type.tensor_type.elem_type
        return guess_dtype(dt)

    def _infer_types(self, cond, named_inputs=None):  # pylint: disable=W0221
        res = self.then_branch._set_type_inference_runtime()
        return tuple([self._pick_type(res, name)
                     for name in self.then_branch.output_names])
