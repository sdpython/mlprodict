# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class Scan(OpRun):

    atts = {
        'body': None,
        'num_scan_inputs': None,
        'scan_input_axes': [],
        'scan_input_directions': [],
        'scan_output_axes': [],
        'scan_output_directions': []
    }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Scan.atts,
                       **options)
        if not hasattr(self.body, 'run'):
            raise RuntimeError("Parameter 'body' must have a method 'run', "
                               "type {}.".format(type(self.body)))
        self.input_directions_ = [0 if i >= len(self.scan_input_directions) else self.scan_input_directions[i]
                                  for i in range(self.num_scan_inputs)]
        max_dir_in = max(self.input_directions_)
        if max_dir_in != 0:
            raise RuntimeError(
                "Scan is not implemented for other output input_direction than 0.")
        self.input_axes_ = [0 if i >= len(self.scan_input_axes) else self.scan_input_axes[i]
                            for i in range(self.num_scan_inputs)]
        max_axe_in = max(self.input_axes_)
        if max_axe_in != 0:
            raise RuntimeError(
                "Scan is not implemented for other input axes than 0.")
        self.input_names = self.body.input_names
        self.output_names = self.body.output_names
        self._run_meth = (self.body.run_in_scan
                          if hasattr(self.body, 'run_in_scan')
                          else self.body.run)

    def _common_run_shape(self, *args):
        num_loop_state_vars = len(args) - self.num_scan_inputs
        num_scan_outputs = len(args) - num_loop_state_vars

        output_directions = [0 if i >= len(self.scan_output_directions) else self.scan_output_directions[i]
                             for i in range(num_scan_outputs)]
        max_dir_out = max(output_directions)
        if max_dir_out != 0:
            raise RuntimeError(
                "Scan is not implemented for other output output_direction than 0.")
        output_axes = [0 if i >= len(self.scan_output_axes) else self.scan_output_axes[i]
                       for i in range(num_scan_outputs)]
        max_axe_out = max(output_axes)
        if max_axe_out != 0:
            raise RuntimeError(
                "Scan is not implemented for other output axes than 0.")

        state_names_in = self.input_names[:self.num_scan_inputs]
        state_names_out = self.output_names[:len(state_names_in)]
        scan_names_in = self.input_names[num_loop_state_vars:]
        scan_names_out = self.output_names[num_loop_state_vars:]
        scan_values = args[num_loop_state_vars:]

        states = args[:num_loop_state_vars]

        return (num_loop_state_vars, num_scan_outputs, output_directions,
                max_dir_out, output_axes, max_axe_out, state_names_in,
                state_names_out, scan_names_in, scan_names_out,
                scan_values, states)

    def _run(self, *args):  # pylint: disable=W0221
        (num_loop_state_vars, num_scan_outputs, output_directions,  # pylint: disable=W0612
         max_dir_out, output_axes, max_axe_out, state_names_in,  # pylint: disable=W0612
         state_names_out, scan_names_in, scan_names_out,  # pylint: disable=W0612
         scan_values, states) = self._common_run_shape(*args)  # pylint: disable=W0612

        max_iter = args[num_loop_state_vars].shape[self.input_axes_[0]]
        results = [[] for _ in scan_names_out]

        for iter in range(max_iter):
            inputs = {}
            for name, value in zip(state_names_in, states):
                inputs[name] = value
            for name, value in zip(scan_names_in, scan_values):
                inputs[name] = value[iter]

            try:
                outputs = self._run_meth(inputs)
            except TypeError as e:
                raise TypeError(
                    "Unable to call 'run' for type '{}'.".format(
                        type(self.body))) from e

            states = [outputs[name] for name in state_names_out]
            for i, name in enumerate(scan_names_out):
                results[i].append(numpy.expand_dims(outputs[name], axis=0))

        for res in results:
            conc = numpy.vstack(res)
            states.append(conc)
        return tuple(states)

    def _infer_shapes(self, *args):  # pylint: disable=W0221
        (num_loop_state_vars, num_scan_outputs, output_directions,  # pylint: disable=W0612
         max_dir_out, output_axes, max_axe_out, state_names_in,  # pylint: disable=W0612
         state_names_out, scan_names_in, scan_names_out,  # pylint: disable=W0612
         scan_values, states) = self._common_run_shape(*args)  # pylint: disable=W0612

        shapes = list(states)

        shape = args[num_loop_state_vars].shape
        if shape is None:
            for sout in scan_values:
                shapes.append(ShapeObject(None, dtype=sout.dtype))
        else:
            max_iter = shape[self.input_axes_[0]]
            for sout in scan_values:
                sc = sout.copy()
                sc[0] = max_iter
                shapes.append(sc)

        return tuple(shapes)
