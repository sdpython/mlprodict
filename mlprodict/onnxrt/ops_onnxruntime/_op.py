# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_onnxruntime*.
"""
import numpy
import onnx.defs
from onnx.helper import make_tensor
from onnxruntime import (
    InferenceSession, SessionOptions, RunOptions, GraphOptimizationLevel)
from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=E0611
    InvalidArgument as OrtInvalidArgument,
    NotImplemented as OrtNotImplemented,
    InvalidGraph as OrtInvalidGraph,
    Fail as OrtFail)
import skl2onnx.algebra.onnx_ops as alg
try:
    import skl2onnx.algebra.custom_ops as alg2
except ImportError:  # pragma: no cover
    # older version of skl2onnx
    alg2 = alg
from ...tools.onnx2py_helper import guess_proto_dtype
from ..optim.graph_schema_helper import (
    get_defined_inputs, get_defined_outputs, proto2vars)


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
        self._schema = _schemas.get(onnx_node.op_type, None)
        if desc is not None:
            if 'atts' in desc:
                for a, b in desc['atts'].items():
                    if not isinstance(b, dict) or 'value' not in b:
                        raise ValueError(  # pragma: no cover
                            "Unexpected value {}.".format(b))
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
                    i += 1  # pragma: no cover
                    new_name = "{}_{}".format(name, i)  # pragma: no cover
                mapping[new_name] = name
                new_inputs.append(new_name)
            else:
                new_inputs.append(name)
                mapping[name] = name
        return mapping, new_inputs

    def _guess_proto_type(self, dtype):
        return guess_proto_dtype(dtype)

    def _init(self, variables=None):
        """
        Initializes the node.

        @param      variables               registered variables created by previous operators

        The current implementation for operator *Scan*
        only works for matrices.
        """
        try:
            self.alg_class = getattr(alg2, 'Onnx' + self.onnx_node.op_type)
        except AttributeError:
            self.alg_class = getattr(alg, 'Onnx' + self.onnx_node.op_type)
        inputs = list(self.onnx_node.input)
        self.mapping, self.inputs = self._name_mapping(inputs)
        self.outputs = list(self.onnx_node.output)

        options = self.options.copy()
        target_opset = options.pop('target_opset', None)
        domain = options.pop('domain', None)
        disable_optimisation = options.pop('disable_optimisation', False)
        ir_version = options.pop('ir_version', None)

        if domain == '' and target_opset < 9:
            # target_opset should be >= 9 not {} for main domain.
            # We assume it was the case when the graph was created.
            pass

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
                                                domain=domain)
                if "dim_value: 0" in str(self.onnx_):
                    raise RuntimeError(  # pragma: no cover
                        "Probable issue as one dimension is null.\n--\n{}".format(
                            self.onnx_))
            except AttributeError as e:  # pragma: no cover
                # older version of skl2onnx
                self.onnx_ = self.inst_.to_onnx(inputs)
                if "dim_value: 0" in str(self.onnx_):
                    raise RuntimeError(
                        "Probable issue as one dimension is null.\n--\n{}".format(
                            self.onnx_)) from e
            forced = False
        elif self.onnx_node.op_type == 'Scan':
            self.inst_ = self.alg_class(
                *self.inputs, output_names=self.outputs,
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
                                            domain=domain)
            if "dim_value: 0" in str(self.onnx_):
                raise RuntimeError(  # pragma: no cover
                    "Probable issue as one dimension is null.\n--\n{}".format(
                        self.onnx_))
            forced = True
        else:
            self.inst_ = self.alg_class(*self.inputs, output_names=self.outputs,
                                        op_version=target_opset, domain=domain,
                                        **options)
            inputs = get_defined_inputs(
                self.inputs, variables, dtype=self.dtype)

            try:
                self.onnx_ = self.inst_.to_onnx(
                    inputs, target_opset=target_opset, domain=domain)
                if "dim_value: 0" in str(self.onnx_):
                    raise RuntimeError(  # pragma: no cover
                        "Probable issue as one dimension is null.\n--\n{}\n---\n{}".format(
                            self.onnx_, inputs))
                forced = False
            except (RuntimeError, ValueError):
                # Let's try again by forcing output types.
                forced = True
                outputs = get_defined_outputs(
                    self.outputs, self.onnx_node, inputs, variables,
                    dtype=self.dtype)
                self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs,
                                                target_opset=target_opset,
                                                domain=domain)
                if "dim_value: 0" in str(self.onnx_):
                    raise RuntimeError(  # pragma: no cover
                        "Probable issue as one dimension is null.\n--\n{}".format(
                            self.onnx_)) from e

        if len(self.onnx_.graph.output) != len(self.outputs):  # pragma: no cover
            # Something is wrong, falls back to default plan.
            forced = True
            outputs = get_defined_outputs(
                self.outputs, self.onnx_node, inputs, variables,
                dtype=self.dtype)
            self.onnx_ = self.inst_.to_onnx(inputs, outputs=outputs,
                                            target_opset=target_opset,
                                            domain=domain)
            if "dim_value: 0" in str(self.onnx_):
                raise RuntimeError(  # pragma: no cover
                    "Probable issue as one dimension is null.\n--\n{}".format(
                        self.onnx_))
        else:
            lo = list(self.onnx_.graph.output)
            outputs = proto2vars(lo)

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
        if ir_version is not None:
            self.onnx_.ir_version = ir_version
        if disable_optimisation:
            sess_options.graph_optimization_level = (  # pragma: no cover
                GraphOptimizationLevel.ORT_DISABLE_ALL)
        try:
            self.sess_ = InferenceSession(
                self.onnx_.SerializeToString(), sess_options=sess_options)
        except (RuntimeError, OrtNotImplemented, OrtInvalidGraph, OrtFail) as e:
            raise RuntimeError("Unable to load node '{}' (output type was {})\n{}".format(
                self.onnx_node.op_type, "guessed" if forced else "inferred",
                self.onnx_)) from e
        self.typed_outputs_ = outputs

    def run(self, *args, **kwargs):
        """
        Should be overwritten.
        """
        inputs = {name: val for name, val in zip(self.inputs, args)}

        try:
            res = self.sess_.run(None, inputs, self.run_options)
        except (RuntimeError, OrtInvalidArgument) as e:  # pragma: no cover
            dtypes = {k: v.dtype for k, v in inputs.items()}
            shapes = {k: v.shape for k, v in inputs.items()}
            exp = [_.name for _ in self.sess_.get_inputs()]
            exp_types = [_.type for _ in self.sess_.get_inputs()]
            raise RuntimeError(
                "Predictions failed. List of inputs: {}, class={}"
                "\ndtypes={}\nshapes={}\nexpected={}\nexpected={}\n"
                "exception={}\n--ONNX--\n{}".format(
                    list(sorted(inputs)), self.alg_class, dtypes,
                    shapes, exp, exp_types, e, self.onnx_)) from e
        return tuple(res)
