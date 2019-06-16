# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_onnxruntime*.
"""
import numpy
import onnx.defs
from onnxruntime import InferenceSession
import skl2onnx.algebra.onnx_ops as alg
from skl2onnx.common.data_types import FloatTensorType


_schemas = {
    schema.name: schema for schema in onnx.defs.get_all_schemas_with_history()}


class OpRunOnnxRuntime:
    """
    Unique operator which calls :epkg:`onnxruntime`
    to compute predictions for one operator.
    """

    def __init__(self, onnx_node, desc=None, **options):
        """
        @param      onnx_node               :epkg:`onnx` node
        @param      desc                    internal representation
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
        self._init()

    def _init(self):
        """
        Initializes the node.
        """
        self.alg_class = getattr(alg, 'Onnx' + self.onnx_node.op_type)
        self.inputs = list(self.onnx_node.input)
        self.outputs = list(self.onnx_node.output)
        self.inst_ = self.alg_class(*self.inputs, output_names=self.outputs,
                                    **self.options)
        inputs = [(name, FloatTensorType()) for name in self.inputs]
        outputs = [(name, FloatTensorType()) for name in self.outputs]
        self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs)
        self.sess_ = InferenceSession(self.onnx_.SerializeToString())

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
