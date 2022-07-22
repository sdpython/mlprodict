# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_whole*.
"""
import json
import numpy


class OnnxWholeSession:
    """
    Runs the prediction for a single :epkg:`ONNX`,
    it lets the runtime handle the graph logic as well.

    :param onnx_data: :epkg:`ONNX` model or data
    :param runtime: runtime to be used, mostly :epkg:`onnxruntime`
    :param runtime_options: runtime options
    :param device: device, a string `cpu`, `cuda`, `cuda:0`...

    .. versionchanged:: 0.8
        Parameter *device* was added.
    """

    def __init__(self, onnx_data, runtime, runtime_options=None, device=None):
        if runtime not in ('onnxruntime1', 'onnxruntime1-cuda'):
            raise NotImplementedError(  # pragma: no cover
                f"runtime '{runtime}' is not implemented.")

        from onnxruntime import (  # delayed
            InferenceSession, SessionOptions, RunOptions,
            GraphOptimizationLevel)
        from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
            Fail as OrtFail, InvalidGraph as OrtInvalidGraph,
            InvalidArgument as OrtInvalidArgument,
            NotImplemented as OrtNotImplemented,
            RuntimeException as OrtRuntimeException)

        onnx_data0 = onnx_data
        if hasattr(onnx_data, 'SerializeToString'):
            onnx_data = onnx_data.SerializeToString()
        if isinstance(runtime_options, SessionOptions):
            sess_options = runtime_options
            session_options = None
            runtime_options = None
        else:
            session_options = (
                None if runtime_options is None
                else runtime_options.get('session_options', None))
            self.runtime = runtime
            sess_options = session_options or SessionOptions()
        self.run_options = RunOptions()
        self.run_options.log_severity_level = 3
        self.run_options.log_verbosity_level = 1

        if session_options is None:
            if runtime_options is not None:
                if runtime_options.get('disable_optimisation', False):
                    sess_options.graph_optimization_level = (  # pragma: no cover
                        GraphOptimizationLevel.ORT_ENABLE_ALL)
                if runtime_options.get('enable_profiling', True):
                    sess_options.enable_profiling = True
                if runtime_options.get('log_severity_level', 2) != 2:
                    v = runtime_options.get('log_severity_level', 2)
                    sess_options.log_severity_level = v
                    self.run_options.log_severity_level = v
        elif runtime_options is not None and 'enable_profiling' in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and enable_profiling cannot be defined at the "
                "same time.")
        elif runtime_options is not None and 'disable_optimisation' in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and disable_optimisation cannot be defined at the "
                "same time.")
        elif runtime_options is not None and 'log_severity_level' in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and log_severity_level cannot be defined at the "
                "same time.")
        providers = ['CPUExecutionProvider']
        if runtime == 'onnxruntime1-cuda':
            providers = ['CUDAExecutionProvider'] + providers
        try:
            self.sess = InferenceSession(onnx_data, sess_options=sess_options,
                                         device=device, providers=providers)
        except (OrtFail, OrtNotImplemented, OrtInvalidGraph,
                OrtInvalidArgument, OrtRuntimeException, RuntimeError) as e:
            from ...plotting.text_plot import onnx_simple_text_plot
            raise RuntimeError(
                "Unable to create InferenceSession due to '{}'\n{}.".format(
                    e, onnx_simple_text_plot(onnx_data0, recursive=True))) from e
        self.output_names = [_.name for _ in self.sess.get_outputs()]

    def run(self, inputs):
        """
        Computes the predictions.

        @param      inputs      dictionary *{variable, value}*
        @return                 list of outputs
        """
        v = next(iter(inputs.values()))
        if isinstance(v, (numpy.ndarray, dict)):
            try:
                return self.sess._sess.run(
                    self.output_names, inputs, self.run_options)
            except ValueError as e:
                raise ValueError(
                    "Issue running inference inputs=%r, expected inputs=%r."
                    "" % (
                        list(sorted(inputs)),
                        [i.name for i in self.sess.get_inputs()])) from e
        try:
            return self.sess._sess.run_with_ort_values(
                inputs, self.output_names, self.run_options)
        except RuntimeError:
            return self.sess._sess.run_with_ort_values(
                {k: v._get_c_value() for k, v in inputs.items()},
                self.output_names, self.run_options)

    @staticmethod
    def process_profiling(js):
        """
        Flattens json returned by onnxruntime profiling.

        :param js: json
        :return: list of dictionaries
        """
        rows = []
        for row in js:
            if 'args' in row and isinstance(row['args'], dict):
                for k, v in row['args'].items():
                    row[f'args_{k}'] = v
                del row['args']
            rows.append(row)
        return rows

    def get_profiling(self):
        """
        Returns the profiling informations.
        """
        prof = self.sess.end_profiling()
        with open(prof, 'r') as f:
            content = f.read()
        js = json.loads(content)
        return OnnxWholeSession.process_profiling(js)
