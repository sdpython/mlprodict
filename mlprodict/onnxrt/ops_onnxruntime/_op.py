# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_onnxruntime*.
"""
import numpy
import onnx.defs
from onnxruntime import InferenceSession
import skl2onnx.algebra.onnx_ops as alg
from ..graph_schema_helper import get_defined_inputs, get_defined_outputs


_schemas = {
    schema.name: schema for schema in onnx.defs.get_all_schemas_with_history()}


class OpRunOnnxRuntime:
    """
    Unique operator which calls :epkg:`onnxruntime`
    to compute predictions for one operator.
    """

    def __init__(self, onnx_node, desc=None, variables=None, **options):
        """
        @param      onnx_node               :epkg:`onnx` node
        @param      desc                    internal representation
        @param      variables               registered variables created by previous operators
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

    def _init(self, variables=None):
        """
        Initializes the node.

        @param      variables               registered variables created by previous operators
        """
        self.alg_class = getattr(alg, 'Onnx' + self.onnx_node.op_type)
        inputs = list(self.onnx_node.input)
        self.mapping, self.inputs = self._name_mapping(inputs)
        self.outputs = list(self.onnx_node.output)
        self.inst_ = self.alg_class(*self.inputs, output_names=self.outputs,
                                    **self.options)
        inputs = get_defined_inputs(self.inputs, variables)
        try:
            self.onnx_ = self.inst_.to_onnx(inputs)
            forced = False
        except RuntimeError:
            # Let's try again by forcing output types.
            forced = True
            outputs = get_defined_outputs(
                self.outputs, self.onnx_node, inputs, variables)
            self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs)
        if len(self.onnx_.graph.output) != self.outputs:
            # Something is wrong, falls back to default plan.
            forced = True
            outputs = get_defined_outputs(
                self.outputs, self.onnx_node, inputs, variables)
            self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs)
        try:
            self.sess_ = InferenceSession(self.onnx_.SerializeToString())
        except RuntimeError as e:
            raise RuntimeError("Unable to load node '{}' (output type was {})\n{}".format(
                self.onnx_node.op_type, "guessed" if forced else "inferred",
                self.onnx_)) from e
        self.typed_outputs_ = outputs

    def run(self, *args, **kwargs):
        """
        Should be overwritten.
        """
        def f32(X):
            if hasattr(X, 'dtype') and X.dtype == numpy.float64:
                return X.astype(numpy.float32)
            else:
                return X

        inputs = {name: f32(val) for name, val in zip(self.inputs, args)}
        res = self.sess_.run(None, inputs)
        return tuple(res)
