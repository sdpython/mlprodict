# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from ._op import OpRun


class _CommonRandom(OpRun):
    """
    Common methods to all random operators.
    """

    def __init__(self, *args, **kwargs):
        OpRun.__init__(self, *args, **kwargs)

    def _dtype(self, *data, dtype_first=False):
        if dtype_first:
            if self.dtype != 0:
                return self.numpy_type
            if len(data) > 0:
                return data[0].dtype
            raise RuntimeError(  # pragma: no cover
                "dtype cannot be None for operator %s, "
                "self.numpy_type=%r, len(data)=%r."
                "" % (self.__class__.__name__,
                      self.numpy_type, len(data)))
        res = None
        if len(data) == 0:
            res = self.numpy_type
        elif self.numpy_type is not None:
            res = self.numpy_type
        elif hasattr(data[0], 'dtype'):
            res = data[0].dtype
        if res is None:
            raise RuntimeError(  # pragma: no cover
                "dtype cannot be None for operator %s, "
                "self.numpy_type=%r, type(data[0])=%r."
                "" % (self.__class__.__name__,
                      self.numpy_type, type(data[0])))
        return res

    def _get_state(self, seed):
        if numpy.isnan(self.seed):
            state = numpy.random.RandomState()
        else:
            state = numpy.random.RandomState(seed=self.seed)
        return state


class Bernoulli(_CommonRandom):

    atts = {'dtype': 0,
            'seed': numpy.nan}

    def __init__(self, onnx_node, desc=None, **options):
        _CommonRandom.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Bernoulli.atts,
                               **options)
        self.numpy_type = (
            TENSOR_TYPE_TO_NP_TYPE[self.dtype] if self.dtype > 0
            else None)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        dtype = self._dtype(x, dtype_first=True)
        state = self._get_state(self.seed)
        res = state.binomial(1, p=x).astype(dtype)
        return (res.astype(dtype), )

    def to_python(self, inputs):
        lines = [
            'numpy_dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]',
            'state = numpy.random.RandomState(seed=seed)',
            f'return state.binomial(1, {inputs[0]}).astype(numpy_dtype)']
        return ("import numpy\nfrom numpy import nan\n"
                "from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE",
                "\n".join(lines))


class RandomUniform(_CommonRandom):

    atts = {'dtype': 1,
            'low': 0.,
            'high': 1.,
            'seed': numpy.nan,
            'shape': []}

    def __init__(self, onnx_node, desc=None, **options):
        _CommonRandom.__init__(self, onnx_node, desc=desc,
                               expected_attributes=RandomUniform.atts,
                               **options)
        if len(self.shape) == 0:
            raise ValueError(  # pragma: no cover
                f"shape cannot be empty for operator {self.__class__.__name__}.")
        self.numpy_type = TENSOR_TYPE_TO_NP_TYPE[self.dtype]

    def _run(self, *args, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(args) != 0:
            raise RuntimeError(  # pragma: no cover
                f"Operator {self.__class__.__name__} cannot have inputs.")
        dtype = self._dtype(*args)
        state = self._get_state(self.seed)
        res = state.rand(*self.shape).astype(dtype)
        res *= (self.high - self.low)
        res += self.low
        return (res.astype(dtype), )

    def to_python(self, inputs):
        lines = [
            'numpy_dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]',
            'state = numpy.random.RandomState(seed=seed)',
            'return (state.rand(*%r).astype(numpy.%s) * (%f - %f)) + %f' % (
                list(self.shape), self.numpy_type, self.high, self.low, self.low)]
        return ("import numpy\nfrom onnx.mapping import TENSOR_TYPE_TO_NP_TYPE",
                "\n".join(lines))


class RandomUniformLike(_CommonRandom):

    atts = {'low': 0.,
            'high': 1.,
            'seed': numpy.nan,
            'dtype': 0}

    def __init__(self, onnx_node, desc=None, **options):
        _CommonRandom.__init__(self, onnx_node, desc=desc,
                               expected_attributes=RandomUniformLike.atts,
                               **options)
        self.numpy_type = (
            None if self.dtype == 0 else TENSOR_TYPE_TO_NP_TYPE[self.dtype])

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        dtype = self._dtype(x)
        state = self._get_state(self.seed)
        res = state.rand(*x.shape).astype(dtype)
        res *= (self.high - self.low)
        res += self.low
        return (res.astype(dtype), )

    def to_python(self, inputs):
        if len(inputs) > 0 and hasattr(inputs[0], 'dtype'):
            dtype = inputs[0].dtype
            shape = inputs[0].shape
        else:
            dtype = self.numpy_type or numpy.float32
            shape = (1, )
        lines = [
            'numpy_dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]',
            'state = numpy.random.RandomState(seed=seed)',
            'return (state.rand(*%r).astype(numpy.%s) * (%f - %f)) + %f' % (
                shape, dtype, self.high, self.low, self.low)]
        return ("import numpy\nfrom onnx.mapping import TENSOR_TYPE_TO_NP_TYPE",
                "\n".join(lines))


class RandomNormal(_CommonRandom):

    atts = {'dtype': 1,
            'mean': 0.,
            'scale': 1.,
            'seed': numpy.nan,
            'shape': []}

    def __init__(self, onnx_node, desc=None, **options):
        _CommonRandom.__init__(self, onnx_node, desc=desc,
                               expected_attributes=RandomNormal.atts,
                               **options)
        if len(self.shape) == 0:
            raise ValueError(  # pragma: no cover
                f"shape cannot be empty for operator {self.__class__.__name__}.")
        self.numpy_type = TENSOR_TYPE_TO_NP_TYPE[self.dtype]

    def _run(self, *args, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(args) != 0:
            raise RuntimeError(  # pragma: no cover
                f"Operator {self.__class__.__name__} cannot have inputs.")
        state = self._get_state(self.seed)
        res = state.randn(*self.shape).astype(self.numpy_type)
        res *= self.scale
        res += self.mean
        return (res.astype(self.numpy_type), )

    def to_python(self, inputs):
        lines = [
            'numpy_dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]',
            'state = numpy.random.RandomState(seed=seed)',
            'return (state.randn(*%r).astype(numpy.%s) * %f) + %f' % (
                list(self.shape), self.numpy_type, self.scale, self.mean)]
        return ("import numpy\nfrom onnx.mapping import TENSOR_TYPE_TO_NP_TYPE",
                "\n".join(lines))


class RandomNormalLike(_CommonRandom):

    atts = {'dtype': 0,
            'mean': 0.,
            'scale': 1.,
            'seed': numpy.nan}

    def __init__(self, onnx_node, desc=None, **options):
        _CommonRandom.__init__(self, onnx_node, desc=desc,
                               expected_attributes=RandomNormalLike.atts,
                               **options)
        self.numpy_type = (
            None if self.dtype == 0 else TENSOR_TYPE_TO_NP_TYPE[self.dtype])

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        dtype = self._dtype(x)
        state = self._get_state(self.seed)
        res = state.randn(*x.shape).astype(dtype)
        res *= self.scale
        res += self.mean
        return (res.astype(dtype), )

    def to_python(self, inputs):
        if len(inputs) > 0 and hasattr(inputs[0], 'dtype'):
            dtype = inputs[0].dtype
            shape = inputs[0].shape
        else:
            dtype = self.numpy_type or numpy.float32
            shape = (1, )
        lines = [
            'numpy_dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]',
            'state = numpy.random.RandomState(seed=seed)',
            'return (state.randn(%r).astype(numpy.%s) * %f) + %f' % (
                shape, dtype, self.scale, self.mean)]
        return ("import numpy\nfrom onnx.mapping import TENSOR_TYPE_TO_NP_TYPE",
                "\n".join(lines))
