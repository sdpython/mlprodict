# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_onnxruntime*.
"""
import numpy
import onnx.defs
from onnxruntime import InferenceSession
import skl2onnx.algebra.onnx_ops as alg
from skl2onnx.common.data_types import (
    DataType,
    FloatTensorType, SequenceType, DictionaryType,
    Int64Type, Int64TensorType
)
from skl2onnx.common.data_types import _guess_type_proto
from skl2onnx.algebra.type_helper import _guess_type


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

    def _init(self, variables=None):
        """
        Initializes the node.

        @param      variables               registered variables created by previous operators
        """
        self.alg_class = getattr(alg, 'Onnx' + self.onnx_node.op_type)
        self.inputs = list(self.onnx_node.input)
        self.outputs = list(self.onnx_node.output)
        # print(self.onnx_node)
        self.inst_ = self.alg_class(*self.inputs, output_names=self.outputs,
                                    **self.options)
        inputs = self.get_defined_inputs(variables)
        try:
            self.onnx_ = self.inst_.to_onnx(inputs)
            forced = False
        except RuntimeError:
            # Let's try again by forcing output types.
            forced = True
            outputs = self.get_defined_outputs()
            self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs)
        if len(self.onnx_.graph.output) != self.outputs:
            # Something is wrong, falls back to default plan.
            forced = True
            outputs = self.get_defined_outputs()
            self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs)
        try:
            self.sess_ = InferenceSession(self.onnx_.SerializeToString())
        except RuntimeError as e:
            raise RuntimeError("Unable to load node '{}' (output type was {})\n{}".format(
                self.onnx_node.op_type, "guessed" if forced else "inferred",
                self.onnx_)) from e
        self.typed_outputs_ = outputs

    def get_defined_inputs(self, variables=None):
        """
        Gets predefined inputs.

        @param      variables               registered variables created by previous operators
        """
        def guess_type_variable(name):
            if variables is None:
                return FloatTensorType()
            elif name in variables:
                ty = variables[name]
                if isinstance(ty, DataType):
                    return variables[name]
                elif isinstance(ty, dict) and 'value' in ty:
                    # constant
                    arr = ty['value']
                    if isinstance(arr, numpy.ndarray) and arr.dtype == numpy.float64:
                        arr = arr.astype(numpy.float32)
                    return _guess_type(arr)
                raise NotImplementedError("Unable to guess type for '{}' form '{}'.".format(
                    name, variables[name]))
            else:
                # Inputs. Let's assume it is a vector of floats.
                return FloatTensorType()

        inputs = [(name, guess_type_variable(name)) for name in self.inputs]
        return inputs

    def get_defined_outputs(self):
        """
        Gets predefined outputs when they cannot be inferred.
        """
        if self.onnx_node.op_type == "ZipMap":
            otype = SequenceType(DictionaryType(
                Int64Type(), FloatTensorType()))
            outputs = [(name, otype) for name in self.outputs]
        elif self.onnx_node.op_type == "Cast" and len(self.outputs) == 1:
            ttyp = _guess_type_proto(self.onnx_node.attribute[0].i, dims=None)
            outputs = [(self.outputs[0], ttyp)]
        elif self.outputs == ['label', 'probability_tensor']:
            # Good chance that's a classifier.
            outputs = [(self.outputs[0], Int64TensorType()),
                       (self.outputs[1], FloatTensorType())]
        else:
            outputs = [(name, FloatTensorType()) for name in self.outputs]
        return outputs

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
