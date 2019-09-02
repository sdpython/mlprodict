# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_onnxruntime*.
"""
import numpy
import onnx.defs
from onnx.helper import make_tensor
from onnx import TensorProto
from onnxruntime import InferenceSession, SessionOptions, RunOptions
import skl2onnx.algebra.onnx_ops as alg
from ..optim.graph_schema_helper import get_defined_inputs, get_defined_outputs


_schemas = {
    schema.name: schema for schema in onnx.defs.get_all_schemas_with_history()}


class OpRunOnnxRuntime:
    """
    Unique operator which calls :epkg:`onnxruntime`
    to compute predictions for one operator.
    """

    def __init__(self, onnx_node, desc=None, variables=None,
                 dtype=None, **options):
        """
        @param      onnx_node               :epkg:`onnx` node
        @param      desc                    internal representation
        @param      variables               registered variables created by previous operators
        @param      dtype                   float computation type
        @param      options                 runtime options
        """
        self._provider = 'onnxruntime'
        self.onnx_node = onnx_node
        self.desc = desc
        self._schema = _schemas[onnx_node.op_type]
        if desc is not None:
            if 'atts' in desc:
                for a, b in desc['atts'].items():
                    if not isinstance(b, dict) or 'value' not in b:
                        raise ValueError("Unexpected value {}.".format(b))
                    options[a] = b['value']

        self.options = options
        self.dtype = dtype
        self._init(variables)

    def _name_mapping(self, inputs):
        mapping = {}
        new_inputs = []
        for name in inputs:
            if name in mapping:
                i = 0
                new_name = "{}_{}".format(name, i)
                while new_name in mapping:
                    i += 1
                    new_name = "{}_{}".format(name, i)
                mapping[new_name] = name
                new_inputs.append(new_name)
            else:
                new_inputs.append(name)
                mapping[name] = name
        return mapping, new_inputs

    def _guess_proto_type(self, dtype):
        if dtype == numpy.float32:
            return TensorProto.FLOAT  # pylint: disable=E1101
        if dtype == numpy.float64:
            return TensorProto.DOUBLE  # pylint: disable=E1101
        raise RuntimeError(
            "Unable to guess type for dtype={}.".format(dtype))

    def _init(self, variables=None):
        """
        Initializes the node.

        @param      variables               registered variables created by previous operators

        The current implementation for operator *Scan*
        only works for matrices.
        """
        self.alg_class = getattr(alg, 'Onnx' + self.onnx_node.op_type)
        inputs = list(self.onnx_node.input)
        self.mapping, self.inputs = self._name_mapping(inputs)
        self.outputs = list(self.onnx_node.output)

        options = self.options.copy()
        target_opset = options.pop('target_opset', None)
        domain = options.pop('domain', None)

        if self.onnx_node.op_type == 'ConstantOfShape':
            for k in options:
                v = options[k]
                if isinstance(v, numpy.ndarray):
                    options[k] = make_tensor(
                        k, self._guess_proto_type(v.dtype),
                        v.shape, v.tolist())

            self.inst_ = self.alg_class(*self.inputs, output_names=self.outputs,
                                        op_version=target_opset, **options)
            inputs = get_defined_inputs(
                self.inputs, variables, dtype=self.dtype)
            try:
                self.onnx_ = self.inst_.to_onnx(inputs, target_opset=target_opset,
                                                dtype=self.dtype, domain=domain)
            except AttributeError:
                # older version of skl2onnx
                self.onnx_ = self.inst_.to_onnx(inputs)
            except TypeError as e:
                import skl2onnx
                raise TypeError(
                    "skl2onnx 1.6.0 is required and you have '{}'.".format(
                        skl2onnx.__version__)) from e
            forced = False
        elif self.onnx_node.op_type == 'Scan':
            self.inst_ = self.alg_class(*self.inputs, output_names=self.outputs,
                                        op_version=target_opset, **options)
            inputs = get_defined_inputs(
                self.inputs, variables, dtype=self.dtype)
            outputs = get_defined_outputs(
                self.outputs, self.onnx_node, inputs, variables,
                dtype=self.dtype)
            inputs = [(name, cl.__class__([None, None]))
                      for (name, cl) in inputs]
            outputs = [(name, cl.__class__([None, None]))
                       for (name, cl) in outputs]
            self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs,
                                            target_opset=target_opset,
                                            dtype=self.dtype, domain=domain)
            forced = True
        else:
            self.inst_ = self.alg_class(*self.inputs, output_names=self.outputs,
                                        op_version=target_opset, domain=domain,
                                        **options)
            inputs = get_defined_inputs(
                self.inputs, variables, dtype=self.dtype)

            try:
                self.onnx_ = self.inst_.to_onnx(
                    inputs, target_opset=target_opset,
                    dtype=self.dtype, domain=domain)
                forced = False
            except (RuntimeError, ValueError):
                # Let's try again by forcing output types.
                forced = True
                outputs = get_defined_outputs(
                    self.outputs, self.onnx_node, inputs, variables,
                    dtype=self.dtype)
                self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs,
                                                target_opset=target_opset,
                                                dtype=self.dtype, domain=domain)

        if len(self.onnx_.graph.output) != self.outputs:
            # Something is wrong, falls back to default plan.
            forced = True
            outputs = get_defined_outputs(
                self.outputs, self.onnx_node, inputs, variables,
                dtype=self.dtype)
            self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs,
                                            target_opset=target_opset,
                                            dtype=self.dtype, domain=domain)

        sess_options = SessionOptions()
        self.run_options = RunOptions()

        try:
            sess_options.session_log_severity_level = 3
            # sess_options.sessions_log_verbosity_level = 0
        except AttributeError:
            # onnxruntime not recent enough.
            pass
        try:
            self.run_options.run_log_severity_level = 3
            # self.run_options.run_log_verbosity_level = 0
        except AttributeError:
            # onnxruntime not recent enough.
            pass
        try:
            self.sess_ = InferenceSession(self.onnx_.SerializeToString(),
                                          sess_options=sess_options)
        except RuntimeError as e:
            raise RuntimeError("Unable to load node '{}' (output type was {})\n{}".format(
                self.onnx_node.op_type, "guessed" if forced else "inferred",
                self.onnx_)) from e
        self.typed_outputs_ = outputs

    def run(self, *args, **kwargs):
        """
        Should be overwritten.
        """
        inputs = {name: val for name, val in zip(self.inputs, args)}

        res = self.sess_.run(None, inputs, self.run_options)
        return tuple(res)
