# coding: utf-8
"""
@file
@brief Speeding up :epkg:`scikit-learn` with :epkg:`onnx`.

.. versionadded:: 0.7
"""
import collections
import inspect
import io
from contextlib import redirect_stdout, redirect_stderr
import numpy
from numpy.testing import assert_almost_equal
from onnx import numpy_helper, helper, load
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import FunctionTransformer
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from ..tools.code_helper import print_code
from ..onnx_conv import to_onnx
from ..onnx_tools.onnx_export import export2numpy
from ..onnx_tools.onnx2py_helper import onnx_model_opsets
from ..onnx_tools.exports.numpy_helper import (
    argmin_use_numpy_select_last_index,
    make_slice)
from .onnx_transformer import OnnxTransformer


class _OnnxPipelineStepSpeedUp(BaseEstimator, OnnxOperatorMixin):
    """
    Speeds up inference by replacing methods *transform* or
    *predict* by a runtime for :epkg:`ONNX`.

    :param estimator: estimator to train
    :param enforce_float32: boolean
        :epkg:`onnxruntime` only supports *float32*,
        :epkg:`scikit-learn` usually uses double floats, this parameter
        ensures that every array of double floats is converted into
        single floats
    :param runtime: string, defined the runtime to use
        as described in @see cl OnnxInference.
    :param target_opset: targetted ONNX opset
    :param conv_options: options for covnersions, see @see fn to_onnx

    .. versionadded:: 0.7
    """

    def __init__(self, estimator, runtime='python', enforce_float32=True,
                 target_opset=None, conv_options=None):
        BaseEstimator.__init__(self)
        self.estimator = estimator
        self.runtime = runtime
        self.enforce_float32 = enforce_float32
        self.target_opset = target_opset
        self.conv_options = conv_options

    def _check_fitted_(self):
        if not hasattr(self, 'onnxrt_'):
            raise AttributeError("Object must be be fit.")

    def _to_onnx(self, fitted_estimator, inputs):
        """
        Converts an estimator inference into :epkg:`ONNX`.

        :param estimator: any estimator following :epkg:`scikit-learn` API
        :param inputs: example of inputs
        :return: ONNX
        """
        opts = self.conv_options or {}
        return to_onnx(
            self.estimator_, inputs, target_opset=self.target_opset,
            **opts)

    def _build_onnx_runtime(self, onx):
        """
        Returns an instance of @see cl OnnxTransformer which
        executes the ONNX graph.

        :param onx: ONNX graph
        :param runtime: runtime type (see @see cl OnnxInference)
        :return: instance of @see cl OnnxInference
        """
        if self.runtime == 'numpy':
            return self._build_onnx_runtime_numpy(onx)
        tr = OnnxTransformer(
            onx, runtime=self.runtime,
            enforce_float32=self.enforce_float32)
        tr.fit()
        return tr

    def _build_onnx_runtime_numpy(self, onx):
        """
        Builds a runtime based on numpy.
        """
        st = io.BytesIO(onx)
        model_onnx = load(st)
        self.numpy_code_ = export2numpy(model_onnx, rename=True)
        opsets = onnx_model_opsets(model_onnx)
        return self._build_onnx_runtime_numpy_compile(opsets)

    def _build_onnx_runtime_numpy_compile(self, opsets):
        try:
            compiled_code = compile(
                self.numpy_code_, '<string>', 'exec')
        except SyntaxError as e:
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, print_code(self.numpy_code_))) from e

        glo = globals().copy()
        loc = {
            'numpy': numpy, 'dict': dict, 'list': list,
            'print': print, 'sorted': sorted,
            'collections': collections, 'inspect': inspect,
            'helper': helper,
            'argmin_use_numpy_select_last_index':
                argmin_use_numpy_select_last_index,
            'make_slice': make_slice}
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(compiled_code, glo, loc)  # pylint: disable=W0122
                except Exception as e:
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              print_code(self.numpy_code_))) from e
        names = [k for k in loc if k.startswith('numpy_')]
        if len(names) != 1:
            raise RuntimeError(
                "Unable to guess which function is the one, names=%r."
                "" % list(sorted(names)))
        fct = loc[names[0]]
        cl = FunctionTransformer(fct, accept_sparse=True)
        cl.op_version = opsets['']
        return cl

    def __getstate__(self):
        state = BaseEstimator.__getstate__(self)
        if 'numpy_code_' in state:
            del state['onnxrt_']
        return state

    def __setstate__(self, state):
        BaseEstimator.__setstate__(self, state)
        if 'numpy_code_' in state:
            st = io.BytesIO(state['onnx_'])
            model_onnx = load(st)
            opsets = onnx_model_opsets(model_onnx)
            self.onnxrt_ = self._build_onnx_runtime_numpy_compile(opsets)

    def fit(self, X, *args, **kwargs):
        """
        Fits the estimator, converts to ONNX.

        :param X: features
        :param args: other arguments
        :param kwargs: fitting options
        """
        if not hasattr(self, 'estimator_'):
            self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, *args, **kwargs)
        if self.enforce_float32:
            X = X.astype(numpy.float32)
        self.onnx_ = self._to_onnx(self.estimator_, X).SerializeToString()
        self.onnxrt_ = self._build_onnx_runtime(self.onnx_)
        return self

    @property
    def op_version(self):
        """
        Returns the opset version.
        """
        self._check_fitted_()
        return self.onnxrt_.op_version

    def onnx_parser(self, scope=None, inputs=None):
        """
        Returns a parser for this model.
        """
        self._check_fitted_()
        if isinstance(self.onnxrt_, FunctionTransformer):
            raise NotImplementedError()
        return self.onnxrt_.onnx_parser(scope, inputs)

    def onnx_shape_calculator(self):
        """
        Returns a shape calculator for this transform.
        """
        self._check_fitted_()
        calc = self.onnxrt_.onnx_shape_calculator()

        def shape_calculator(operator):
            return calc(operator)

        return shape_calculator

    def onnx_converter(self):
        """
        Returns a converter for this transform.
        """
        self._check_fitted_()
        conv = self.onnxrt_.onnx_converter()

        def converter(scope, operator, container):
            op = operator.raw_operator
            onnx_model = op.onnxrt_.onnxrt_.obj
            conv(scope, operator, container, onnx_model=onnx_model)

        return converter


class OnnxSpeedUpTransformer(TransformerMixin,
                             _OnnxPipelineStepSpeedUp):
    """
    Trains with :epkg:`scikit-learn`, transform with :epkg:`ONNX`.

    :param estimator: estimator to train
    :param enforce_float32: boolean
        :epkg:`onnxruntime` only supports *float32*,
        :epkg:`scikit-learn` usually uses double floats, this parameter
        ensures that every array of double floats is converted into
        single floats
    :param runtime: string, defined the runtime to use
        as described in @see cl OnnxInference.
    :param target_opset: targetted ONNX opset
    :param conv_options: conversion options, see @see fn to_onnx

    .. versionadded:: 0.7
    """

    def __init__(self, estimator, runtime='python', enforce_float32=True,
                 target_opset=None, conv_options=None):
        _OnnxPipelineStepSpeedUp.__init__(
            self, estimator, runtime=runtime, enforce_float32=enforce_float32,
            target_opset=target_opset, conv_options=conv_options)

    def fit(self, X, y=None, sample_weight=None):  # pylint: disable=W0221
        """
        Trains based estimator.
        """
        if sample_weight is None:
            _OnnxPipelineStepSpeedUp.fit(self, X, y)
        else:
            _OnnxPipelineStepSpeedUp.fit(
                self, X, y, sample_weight=sample_weight)
        return self

    def transform(self, X):
        """
        Transforms with *ONNX*.

        :param X: features
        :return: transformed features
        """
        return self.onnxrt_.transform(X)

    def raw_transform(self, X):
        """
        Transforms with *scikit-learn*.

        :param X: features
        :return: transformed features
        """
        return self.estimator_.transform(X)

    def assert_almost_equal(self, X, **kwargs):
        """
        Checks that ONNX and scikit-learn produces the same
        outputs.
        """
        expected = self.raw_transform(X)
        got = self.transform(X)
        assert_almost_equal(expected, got, **kwargs)
