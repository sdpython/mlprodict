"""
@file
@brief Micro runtime for ONNX.

.. versionadded:: 0.6
"""
import numpy
from ..onnx_tools.onnx2py_helper import _var_as_dict


class OnnxMicroRuntime:
    """
    Implements a micro runtime for ONNX graphs.
    It does not implements all the operator types.

    :param model_onnx: ONNX model

    .. runpython::
        :showcode:

        import pprint
        import numpy
        from mlprodict.onnxrt.onnx_micro_runtime import OnnxMicroRuntime
        from mlprodict.npy.xop import loadop

        OnnxAdd = loadop('Add')

        dtype = numpy.float32
        opset = 15
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=opset)
        cop4 = OnnxAdd(cop, numpy.array([2], dtype=dtype), op_version=opset,
                       output_names=['Y'])
        model_def = cop4.to_onnx({'X': x}, target_opset=opset)
        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        pprint.pprint(out)
    """

    def __init__(self, model_onnx):
        if not hasattr(model_onnx, 'graph'):
            raise TypeError(
                f"model_onnx is not an ONNX graph but {type(model_onnx)!r}.")
        self.model_onnx = model_onnx

    @property
    def input_names(self):
        "Returns input names."
        return [i.name for i in self.model_onnx.graph.input]

    @property
    def output_names(self):
        "Returns output names."
        return [i.name for i in self.model_onnx.graph.output]

    def run(self, inputs):
        """
        Computes the outputs of the graph.

        :param inputs: dictionary
        :return: all intermediates results and output as a dictionary
        """
        if not isinstance(inputs, dict):
            raise TypeError(
                f"inputs must be a dictionary not {type(inputs)!r}.")
        results = inputs.copy()

        for init in self.model_onnx.graph.initializer:
            name = init.name
            mat = _var_as_dict(init)['value']
            results[name] = mat

        for node in self.model_onnx.graph.node:
            op_type = node.op_type
            inp = [results[n] for n in node.input]
            meth_name = f"_op_{op_type.lower()}"
            if not hasattr(self, meth_name):
                raise NotImplementedError(
                    f"OnnxMicroRuntime does not implement operator {op_type!r}.")
            kwargs = {}
            for at in node.attribute:
                var = _var_as_dict(at)
                kwargs[at.name] = var['value']
            out = getattr(self, meth_name)(*inp, **kwargs)
            for n, o in zip(node.output, out):
                results[n] = o

        return results

    ########################
    # Runtime for operators
    ########################

    def _op_abs(self, x):
        "Runtime for operator :epkg:`Op:Abs`."
        return (numpy.abs(x), )

    def _op_add(self, x, y):
        "Runtime for operator :epkg:`Op:Add`."
        return (x + y, )

    def _op_concat(self, *args, axis=None):
        "Runtime for operator :epkg:`Op:Concat`."
        def _preprocess(a, axis):
            if axis >= len(a.shape):
                new_shape = a.shape + (1, ) * (axis + 1 - len(a.shape))
                return a.reshape(new_shape)
            return a

        targs = tuple(_preprocess(a, axis) for a in args)
        return (numpy.concatenate(targs, axis), )

    def _op_gemm(self, a, b, c=None, alpha=None, beta=None,
                 transA=False, transB=False):
        "Runtime for operator :epkg:`Op:Gemm`."

        def _gemm00(a, b, c, alpha, beta):
            o = numpy.dot(a, b) * alpha
            if beta != 0:
                o += c * beta
            return o

        def _gemm01(a, b, c, alpha, beta):
            o = numpy.dot(a, b.T) * alpha
            if beta != 0:
                o += c * beta
            return o

        def _gemm10(a, b, c, alpha, beta):
            o = numpy.dot(a.T, b) * alpha
            if beta != 0:
                o += c * beta
            return o

        def _gemm11(a, b, c, alpha, beta):
            o = numpy.dot(a.T, b.T) * alpha
            if beta != 0:
                o += c * beta
            return o

        if not isinstance(transA, (int, bool, numpy.int64)):
            raise TypeError(  # pragma: no cover
                f"Unexpected type for transA: {type(transA)!r}.")
        if not isinstance(transB, (int, bool, numpy.int64)):
            raise TypeError(  # pragma: no cover
                f"Unexpected type for transA: {type(transB)!r}.")
        if transA:
            fct = _gemm11 if transB else _gemm10
        else:
            fct = _gemm01 if transB else _gemm00
        return (fct(a, b, c, alpha=alpha, beta=beta), )

    def _op_gather(self, x, indices, axis=None):
        "Runtime for operator :epkg:`Op:Gather`."
        if not x.flags['C_CONTIGUOUS']:
            x = numpy.ascontiguousarray(x)
        if not indices.flags['C_CONTIGUOUS']:
            indices = indices.ascontiguousarray()
        return (numpy.take(x, indices, axis=axis), )

    def _op_identity(self, x):
        "Runtime for operator :epkg:`Op:Identity`."
        return (x, )

    def _op_matmul(self, x, y):
        "Runtime for operator :epkg:`Op:MatMul`."
        return (numpy.matmul(x, y), )

    def _op_max(self, *inps):
        "Runtime for operator :epkg:`Op:Max`."
        return (numpy.maximum(*inps), )

    def _op_mul(self, x, y):
        "Runtime for operator :epkg:`Op:Mul`."
        return (x * y, )

    def _op_reduceprod(self, data, axes=None, keepdims=None):
        "Runtime for operator :epkg:`Op:ReduceProd`."
        if axes is not None and not isinstance(axes, int):
            if isinstance(axes, numpy.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes) if len(axes) > 0 else None
        return (numpy.prod(data, axis=axes,
                           keepdims=keepdims,
                           dtype=data.dtype), )

    def _op_reducesum(self, data, axes, keepdims=None,
                      noop_with_empty_axes=None):
        "Runtime for operator :epkg:`Op:ReduceSum`."
        if axes is None and noop_with_empty_axes:
            return (data, )
        if axes is not None and not isinstance(axes, int):
            if isinstance(axes, numpy.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes) if len(axes) > 0 else None
        return (numpy.sum(data, axis=axes,
                          keepdims=keepdims,
                          dtype=data.dtype), )

    def _op_reshape(self, x, shape):
        "Runtime for operator :epkg:`Op:Reshape`."
        return (x.reshape(shape), )

    def _op_shape(self, x):
        "Runtime for operator :epkg:`Op:Shape`."
        return (numpy.array(list(x.shape), dtype=numpy.int64), )

    def _op_squeeze(self, x, axes=None):
        "Runtime for operator :epkg:`Op:Squeeze`."
        if axes is None:
            return (x, )
        if hasattr(axes, '__iter__'):
            return (numpy.squeeze(x, axis=tuple(axes)), )
        return (numpy.squeeze(x, axis=axes), )

    def _op_transpose(self, x, perm=None):
        "Runtime for operator :epkg:`Op:Transpose`."
        return (numpy.transpose(x, perm), )

    def _op_unsqueeze(self, x, axes=None):
        "Runtime for operator :epkg:`Op:Unsqueeze`."
        if axes is None:
            return (x, )
        if hasattr(axes, '__iter__'):
            return (numpy.expand_dims(x, axis=tuple(axes)), )
        return (numpy.expand_dims(x, axis=axes), )
