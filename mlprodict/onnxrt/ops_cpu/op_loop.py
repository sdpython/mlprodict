# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Loop(OpRun):

    atts = {
        'body': None,
    }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Loop.atts,
                       **options)
        if not hasattr(self.body, 'run'):
            raise RuntimeError("Parameter 'body' must have a method 'run', "
                               "type {}.".format(type(self.body)))

        self._run_meth = (self.body.run_in_scan
                          if hasattr(self.body, 'run_in_scan')
                          else self.body.run)

    def _run(self, M, cond, v_initial, *args):  # pylint: disable=W0221
        inputs = {name: None for name in self.body.input_names}
        inputs[self.body.input_names[0]] = cond
        inputs[self.body.input_names[1]] = v_initial
        cond_name = self.body.output_names[0]
        if len(args) > 0:
            begin = len(self.body.input_names) - len(args)
            for name, val in zip(self.body.input_names[begin:], args):
                inputs[name] = val
        it = 0
        while cond and it < M:
            outputs = self._run_meth_then(inputs)
            cond = outputs[cond_name]
            for i, o in zip(self.body.input_names[2:],
                            self.body.output_names[2:]):
                inputs[i] = outputs[o]
            it += 1
        if it == 0:
            outputs = {self.body.output_names[1]: cond}
            for i, o in zip(self.body.input_names[2:],
                            self.body.output_names[2:]):
                outputs[o] = inputs[i]
        for o in self.body.output_names:
            if o not in outputs:
                outputs[o] = numpy.empty(shape=tuple())
        return tuple([outputs[name] for name in self.body.output_names[1:]])

    def _infer_shapes(self, M, cond, v_initial, *args):  # pylint: disable=W0221
        res = self.body._set_shape_inference_runtime()
        return tuple([res[name] for name in self.body.output_names[1:]])
