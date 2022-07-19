# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.

.. versionadded:: 0.7
"""
import numpy
from ._op import OpRun


class Loop(OpRun):

    atts = {'body': None}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Loop.atts,
                       **options)
        if not hasattr(self.body, 'run'):
            raise RuntimeError(  # pragma: no cover
                f"Parameter 'body' must have a method 'run', type {type(self.body)}.")

        self._run_meth = (self.body.run_in_scan
                          if hasattr(self.body, 'run_in_scan')
                          else self.body.run)
        self.additional_inputs = self.body.static_inputs

    def need_context(self):
        """
        The operator Loop needs to know all results produced
        so far as the loop may silently access one of them.
        Some information are not always referred in the list of inputs
        (kind of static variables).
        """
        return len(self.additional_inputs) > 0

    def _run(self, M, cond, v_initial, *args, callback=None, context=None,  # pylint: disable=W0221
             attributes=None, verbose=0, fLOG=None):
        loop_inputs = self.body.input_names
        inputs = {name: None for name in loop_inputs}
        inputs[loop_inputs[2]] = v_initial
        cond_name = self.body.output_names[0]
        if len(args) > 0:
            begin = len(loop_inputs) - len(args)
            all_inputs = loop_inputs[begin:]
            for name, val in zip(all_inputs, args):
                inputs[name] = val
        if len(self.additional_inputs) > 0:
            if context is None:
                raise RuntimeError(
                    "Additional inputs %r are missing and context is None."
                    "" % (self.additional_inputs, ))
            for a in self.additional_inputs:
                if a in context:
                    inputs[a] = context[a]
                else:
                    raise RuntimeError(
                        "Additional inputs %r not found in context\n%s." % (
                            a, "\n".join(sorted(map(str, context)))))

        it = 0
        while cond and it < M:
            inputs[self.body.input_names[0]] = numpy.array(it, dtype=M.dtype)
            inputs[self.body.input_names[1]] = cond
            outputs = self._run_meth(inputs)
            cond = outputs[cond_name]
            if cond is None:
                raise RuntimeError(
                    f"condition {cond_name!r} returned by the subgraph cannot be None.")
            for i, o in zip(self.body.input_names[2:],
                            self.body.output_names[1:]):
                inputs[i] = outputs[o]
            if callback is not None:
                callback(inputs, context=context)
            it += 1

        if it == 0:
            outputs = {self.body.output_names[1]: cond}
            for i, o in zip(self.body.input_names[2:],
                            self.body.output_names[1:]):
                outputs[o] = inputs[i]
        for o in self.body.output_names:
            if o not in outputs:
                outputs[o] = numpy.empty(shape=tuple())
        res = tuple([outputs[name] for name in self.body.output_names[1:]])
        if any(r is None for r in res):
            raise TypeError(  # pragma: no cover
                "Operator Loop produces a None value.")
        return res
