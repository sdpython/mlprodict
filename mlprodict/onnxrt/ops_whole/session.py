# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_whole*.
"""
import json
from io import BytesIO
import onnx
from ...tools.ort_wrapper import (
    InferenceSession, SessionOptions, RunOptions,
    GraphOptimizationLevel, OrtFail,
    OrtInvalidGraph, OrtInvalidArgument,
    OrtNotImplemented, OrtRuntimeException)
from ...tools.asv_options_helper import display_onnx


class OnnxWholeSession:
    """
    Runs the prediction for a single :epkg:`ONNX`,
    it lets the runtime handle the graph logic as well.
    """

    def __init__(self, onnx_data, runtime, runtime_options=None):
        """
        @param      onnx_data       :epkg:`ONNX` model or data
        @param      runtime         runtime to be used,
                                    mostly :epkg:`onnxruntime`
        @param      runtime_options runtime options
        """
        if runtime != 'onnxruntime1':
            raise NotImplementedError(  # pragma: no cover
                "runtime '{}' is not implemented.".format(runtime))
        if hasattr(onnx_data, 'SerializeToString'):
            onnx_data = onnx_data.SerializeToString()
        session_options = (
            None if runtime_options is None
            else runtime_options.get('session_options', None))
        self.runtime = runtime
        sess_options = session_options or SessionOptions()
        self.run_options = RunOptions()

        if session_options is None:
            try:
                sess_options.sessions_log_verbosity_level = 0
            except AttributeError:  # pragma: no cover
                # onnxruntime not recent enough.
                pass
            try:
                self.run_options.run_log_verbosity_level = 0
            except AttributeError:  # pragma: no cover
                # onnxruntime not recent enough.
                pass
            if runtime_options is not None:
                if runtime_options.get('disable_optimisation', False):
                    sess_options.graph_optimization_level = (  # pragma: no cover
                        GraphOptimizationLevel.ORT_ENABLE_ALL)
                if runtime_options.get('enable_profiling', True):
                    sess_options.enable_profiling = True
        elif 'enable_profiling' in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and enable_profiling cannot be defined at the "
                "same time.")
        elif 'disable_optimisation' in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and disable_optimisation cannot be defined at the "
                "same time.")
        try:
            self.sess = InferenceSession(onnx_data, sess_options=sess_options)
        except (OrtFail, OrtNotImplemented, OrtInvalidGraph,
                OrtInvalidArgument, OrtRuntimeException, RuntimeError) as e:
            raise RuntimeError(
                "Unable to create InferenceSession due to '{}'\n{}.".format(
                    e, display_onnx(onnx.load(BytesIO(onnx_data))))) from e

    def run(self, inputs):
        """
        Computes the predictions.

        @param      inputs      dictionary *{variable, value}*
        @return                 list of outputs
        """
        return self.sess.run(None, inputs, self.run_options)

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
                    row['args_%s' % k] = v
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
