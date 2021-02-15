"""
@file
@brief Implements :epkg:`numpy` functions with onnx and a runtime.
"""
import inspect
from typing import Any
try:
    from numpy.typing import NDArray as typing_NDArray
except ImportError:
    from nptyping import NDArray as typing_NDArray
from ..onnxrt import OnnxInference
from .onnx_variable import OnnxVar


class OnnxNumpyCompiler:
    """
    Implements a class which runs onnx graph.

    :param fct: a function with annotations which returns an ONNX graph,
        it can also be an ONNX graph.
    :param op_version: :epkg:`ONNX` opset to use, None
        for the latest one
    :param runtime: runtime to choose to execute the onnx graph,
        `python`, `onnxruntime`, `onnxruntime1`
    """
    NDArray = typing_NDArray

    def __init__(self, fct, op_version=None, runtime=None):
        if hasattr(fct, 'SerializeToString'):
            self.fct_ = None
            self.onnx_ = fct
        else:
            self.fct_ = fct
            if not inspect.isfunction(fct):
                raise TypeError(
                    "Unexpected type for fct=%r, it must be function." % fct)
            self.onnx_ = None
            self.onnx_ = self._to_onnx(op_version=op_version)
        self.runtime_ = self._build_runtime(op_version=op_version,
                                            runtime=runtime)

    def __repr__(self):
        "usual"
        if self.fct_ is not None:
            return "%s(%s)" % (self.__class__.__name__, repr(self.fct_))
        if self.onnx_ is not None:
            return "%s(%s)" % (self.__class__.__name__, "... ONNX ... ")
        raise NotImplementedError(
            "fct_ and onnx_ are empty.")

    def _to_onnx_shape(self, shape):
        if shape is Any:
            shape = None
        return shape

    def _to_onnx_dtype(self, dtype, shape):
        from skl2onnx.common.data_types import _guess_numpy_type
        return _guess_numpy_type(dtype, shape)

    def _parse_annotation(self):
        """
        Returns the annotations for function `fct_`.
        """
        args = self.fct_.__code__.co_varnames
        annotations = self.fct_.__annotations__
        inputs = []
        outputs = []
        for a in args:
            if a == "op_version":
                continue
            if a not in annotations:
                raise RuntimeError(
                    "Unable to find annotation for argument %r." % a)
            ann = annotations[a]
            shape, dtype = ann.__args__
            shape = self._to_onnx_shape(shape)
            dtype = self._to_onnx_dtype(dtype, shape)
            inputs.append((a, dtype))
        ret = annotations['return']
        shape, dtype = ret.__args__
        shape = self._to_onnx_shape(shape)
        dtype = self._to_onnx_dtype(dtype, shape)
        outputs.append(('y', dtype))
        return inputs, outputs

    def _to_onnx(self, op_version=None):
        """
        Returns the onnx graph produced by function `fct_`.
        """
        if self.onnx_ is None and self.fct_ is not None:
            inputs, outputs = self._parse_annotation()
            names_in = [oi[0] for oi in inputs]
            names_out = [oi[0] for oi in outputs]
            names_var = [OnnxVar(n) for n in names_in]
            if 'op_version' in self.fct_.__code__.co_varnames:
                onx_algebra = self.fct_(*names_in, op_version=op_version)
            else:
                onx_var = self.fct_(*names_var)
                onx_algebra = onx_var.to_algebra(op_version=op_version)
            if isinstance(onx_algebra, str):
                raise RuntimeError(  # pragma: no cover
                    "Unexpected str type %r." % onx_algebra)
            if isinstance(onx_algebra, tuple):
                raise NotImplementedError(
                    "Not implemented when the function returns multiple results.")
            if hasattr(onx_algebra, 'to_onnx'):
                # skl2onnx algebra
                onx_algebra.output_names = names_out
                onx = onx_algebra.to_onnx(inputs=inputs,
                                          target_opset=op_version,
                                          outputs=outputs)
                self.onnx_ = onx

        if self.onnx_ is None:
            raise RuntimeError(
                "Unable to get the ONNX graph.")
        return self.onnx_

    def _build_runtime(self, op_version=None, runtime=None):
        """
        Creates the runtime for the :epkg:`ONNX` graph.

        :param op_version: :epkg:`ONNX` opset to use, None
            for the latest one
        :param runtime: runtime to choose to execute the onnx graph,
            `python`, `onnxruntime`, `onnxruntime1`
        """
        onx = self._to_onnx(op_version=op_version)
        inputs, outputs = self._parse_annotation()
        if runtime != 'onnxruntime':
            rt = OnnxInference(onx, runtime=runtime)
            self.rt_ = rt

            def _fct_oi_(*args, inputs=tuple(inputs), outputs=tuple(outputs),
                         rt=self.rt_):
                if len(args) != len(inputs):
                    raise RuntimeError(
                        "Unexpected number of inputs %d instead of %d." % (
                            len(args), len(inputs)))
                inp = {k[0]: a for k, a in zip(inputs, args)}
                out = rt.run(inp)
                return tuple([out[o[0]] for o in outputs])

            self.rt_fct_ = _fct_oi_
        else:
            from onnxruntime import InferenceSession
            rt = InferenceSession(onx.SerializeToString())
            self.rt_ = rt

            def _fct_oi_(*args, inputs=tuple(inputs), outputs=tuple(outputs),
                         rt=self.rt_):
                if len(args) != len(inputs):
                    raise RuntimeError(
                        "Unexpected number of inputs %d instead of %d." % (
                            len(args), len(inputs)))
                inp = {k[0]: a for k, a in zip(inputs, args)}
                out = rt.run(None, inp)
                return tuple(out)

            self.rt_fct_ = _fct_oi_
        return self.rt_

    def __call__(self, *args):
        """
        Executes the function and returns the results.

        :param args: arguments
        :return: results
        """
        res = self.rt_fct_(*args)
        if len(res) == 1:
            return res[0]
        return res
