# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ...onnx_tools.onnx2py_helper import guess_dtype
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

    def _run(self, cond, named_inputs=None, context=None,  # pylint: disable=W0221
             attributes=None, verbose=0, fLOG=None):
        if cond is None:
            raise RuntimeError(  # pragma: no cover
                "cond cannot be None")
        if named_inputs is None:
            named_inputs = {}
        if len(self.then_branch.input_names) > 0:
            if len(context) == 0:
                raise RuntimeError(  # pragma: no cover
                    "named_inputs is empty but the graph needs {}, "
                    "sub-graphs for node If must not have any inputs.".format(
                        self.then_branch.input_names))
            for k in self.then_branch.input_names:
                if k not in context:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to find named input '{}' in\n{}.".format(
                            k, "\n".join(sorted(context))))
        if len(self.else_branch.input_names) > 0:
            if len(context) == 0:
                raise RuntimeError(  # pragma: no cover
                    "context is empty but the graph needs {}.".format(
                        self.then_branch.input_names))
            for k in self.else_branch.input_names:
                if k not in context:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to find named input '{}' in\n{}.".format(
                            k, "\n".join(sorted(context))))

        # then_local_inputs = set(self.local_inputs(self.then_branch.obj.graph))
        # else_local_inputs = set(self.local_inputs(self.else_branch.obj.graph))
        # self.additional_inputs = list(
        #     set(self.additional_inputs).union(then_local_inputs.union(else_local_inputs)))
        # for n in self.additional_inputs:
        #     self.then_branch.global_index(n)
        #     self.else_branch.global_index(n)

        if len(cond.shape) > 0:
            if all(cond):
                if verbose > 0 and fLOG is not None:
                    fLOG(  # pragma: no cover
                        f'  -- then> {list(context)!r}')
                outputs = self._run_meth_then(named_inputs, context=context,
                                              attributes=attributes,
                                              verbose=verbose, fLOG=fLOG)
                if verbose > 0 and fLOG is not None:
                    fLOG('  -- then<')
                final = tuple([outputs[name]
                              for name in self.then_branch.output_names])
                branch = 'then'
            else:
                if verbose > 0 and fLOG is not None:
                    fLOG(  # pragma: no cover
                        f'  -- else> {list(context)!r}')
                outputs = self._run_meth_else(named_inputs, context=context,
                                              attributes=attributes,
                                              verbose=verbose, fLOG=fLOG)
                if verbose > 0 and fLOG is not None:
                    fLOG('  -- else<')  # pragma: no cover
                final = tuple([outputs[name]
                              for name in self.else_branch.output_names])
                branch = 'else'
        elif cond:
            if verbose > 0 and fLOG is not None:
                fLOG(  # pragma: no cover
                    f'  -- then> {list(context)!r}')
            outputs = self._run_meth_then(named_inputs, context=context,
                                          attributes=attributes,
                                          verbose=verbose, fLOG=fLOG)
            if verbose > 0 and fLOG is not None:
                fLOG('  -- then<')  # pragma: no cover
            final = tuple([outputs[name]
                          for name in self.then_branch.output_names])
            branch = 'then'
        else:
            if verbose > 0 and fLOG is not None:
                fLOG(  # pragma: no cover
                    f'  -- else> {list(context)!r}')
            outputs = self._run_meth_else(named_inputs, context=context,
                                          attributes=attributes,
                                          verbose=verbose, fLOG=fLOG)
            if verbose > 0 and fLOG is not None:
                fLOG('  -- else<')  # pragma: no cover
            final = tuple([outputs[name]
                          for name in self.else_branch.output_names])
            branch = 'else'

        if len(final) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Operator If ({self.onnx_node.name!r}) does not have any output.")
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

    def _pick_type(self, res, name):
        if name in res:
            return res[name]
        out = {o.name: o for o in self.then_branch.obj.graph.output}
        if name not in out:
            raise ValueError(  # pragma: no cover
                "Unable to find name=%r in %r or %r." % (
                    name, list(sorted(res)), list(sorted(out))))
        dt = out[name].type.tensor_type.elem_type
        return guess_dtype(dt)
