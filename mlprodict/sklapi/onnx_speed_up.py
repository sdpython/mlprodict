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
import scipy.special as scipy_special
import scipy.spatial.distance as scipy_distance
from onnx import helper, load
from sklearn.base import (
    BaseEstimator, clone,
    TransformerMixin, RegressorMixin, ClassifierMixin,
    ClusterMixin)
from sklearn.preprocessing import FunctionTransformer
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from ..tools.code_helper import print_code
from ..tools.asv_options_helper import get_opset_number_from_onnx
from ..onnx_tools.onnx_export import export2numpy
from ..onnx_tools.onnx2py_helper import (
    onnx_model_opsets, _var_as_dict, to_skl2onnx_type)
from ..onnx_tools.exports.numpy_helper import (
    array_feature_extrator,
    argmax_use_numpy_select_last_index,
    argmin_use_numpy_select_last_index,
    make_slice)
from ..onnx_tools.exports.skl2onnx_helper import add_onnx_graph
from ..onnx_conv import to_onnx
from .onnx_transformer import OnnxTransformer


class _OnnxPipelineStepSpeedup(BaseEstimator, OnnxOperatorMixin):
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
    :param conv_options: options for conversions, see @see fn to_onnx
    :param nopython: used by :epkg:`numba` jitter

    Attributes created by method *fit*:

    * `estimator_`: cloned and trained version of *estimator*
    * `onnxrt_`: objet of type @see cl OnnxInference,
        :epkg:`sklearn:preprocessing:FunctionTransformer`
    * `numpy_code_`: python code equivalent to the inference
        method if the runtime is `'numpy'` or `'numba'`
    * `onnx_io_names_`: dictionary, additional information
        if the runtime is `'numpy'` or `'numba'`

    .. versionadded:: 0.7
    """

    def __init__(self, estimator, runtime='python', enforce_float32=True,
                 target_opset=None, conv_options=None, nopython=True):
        BaseEstimator.__init__(self)
        self.estimator = estimator
        self.runtime = runtime
        self.enforce_float32 = enforce_float32
        self.target_opset = target_opset
        self.conv_options = conv_options
        self.nopython = nopython

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
        return to_onnx(
            self.estimator_, inputs, target_opset=self.target_opset,
            options=self.conv_options)

    def _build_onnx_runtime(self, onx):
        """
        Returns an instance of @see cl OnnxTransformer which
        executes the ONNX graph.

        :param onx: ONNX graph
        :param runtime: runtime type (see @see cl OnnxInference)
        :return: instance of @see cl OnnxInference
        """
        if self.runtime in ('numpy', 'numba'):
            return self._build_onnx_runtime_numpy(onx)
        tr = OnnxTransformer(
            onx, runtime=self.runtime,
            enforce_float32=self.enforce_float32)
        tr.fit()
        return tr

    def _build_onnx_runtime_numpy(self, onx):
        """
        Builds a runtime based on numpy.
        Exports the ONNX graph into python code
        based on numpy and then dynamically compiles
        it with method @see me _build_onnx_runtime_numpy_compile.
        """
        model_onnx = load(io.BytesIO(onx))
        self.onnx_io_names_ = {'inputs': [], 'outputs': []}
        for inp in model_onnx.graph.input:  # pylint: disable=E1101
            d = _var_as_dict(inp)
            self.onnx_io_names_['inputs'].append((d['name'], d['type']))
        for inp in model_onnx.graph.output:  # pylint: disable=E1101
            d = _var_as_dict(inp)
            self.onnx_io_names_['outputs'].append((d['name'], d['type']))
        self.onnx_io_names_['skl2onnx_inputs'] = [
            to_skl2onnx_type(d[0], d[1]['elem'], d[1]['shape'])
            for d in self.onnx_io_names_['inputs']]
        self.onnx_io_names_['skl2onnx_outputs'] = [
            to_skl2onnx_type(d[0], d[1]['elem'], d[1]['shape'])
            for d in self.onnx_io_names_['outputs']]
        self.numpy_code_ = export2numpy(model_onnx, rename=True)
        opsets = onnx_model_opsets(model_onnx)
        return self._build_onnx_runtime_numpy_compile(opsets)

    def _build_onnx_runtime_numpy_compile(self, opsets):
        """
        Second part of @see me _build_onnx_runtime_numpy.
        """
        try:
            compiled_code = compile(
                self.numpy_code_, '<string>', 'exec')
        except SyntaxError as e:  # pragma: no cover
            raise AssertionError(
                "Unable to compile a script due to %r. "
                "\n--CODE--\n%s"
                "" % (e, print_code(self.numpy_code_))) from e

        glo = globals().copy()
        loc = {
            'numpy': numpy, 'dict': dict, 'list': list,
            'print': print, 'sorted': sorted,
            'collections': collections, 'inspect': inspect,
            'helper': helper, 'scipy_special': scipy_special,
            'scipy_distance': scipy_distance,
            'array_feature_extrator': array_feature_extrator,
            'argmin_use_numpy_select_last_index':
                argmin_use_numpy_select_last_index,
            'argmax_use_numpy_select_last_index':
                argmax_use_numpy_select_last_index,
            'make_slice': make_slice}
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out):
            with redirect_stderr(err):
                try:
                    exec(compiled_code, glo, loc)  # pylint: disable=W0122
                except Exception as e:  # pragma: no cover
                    raise AssertionError(
                        "Unable to execute a script due to %r. "
                        "\n--OUT--\n%s\n--ERR--\n%s\n--CODE--\n%s"
                        "" % (e, out.getvalue(), err.getvalue(),
                              print_code(self.numpy_code_))) from e
        names = [k for k in loc if k.startswith('numpy_')]
        if len(names) != 1:
            raise RuntimeError(  # pragma: no cover
                "Unable to guess which function is the one, names=%r."
                "" % list(sorted(names)))
        fct = loc[names[0]]
        if self.runtime == 'numba':
            from numba import jit
            jitter = jit(nopython=self.nopython)
            fct = jitter(fct)
        cl = FunctionTransformer(fct, accept_sparse=True)
        cl.op_version = opsets.get('', get_opset_number_from_onnx())
        return cl

    def __getstate__(self):
        """
        :epkg:`pickle` does not support functions.
        This method removes any link to function
        when the runtime is `'numpy'`.
        """
        state = BaseEstimator.__getstate__(self)
        if 'numpy_code_' in state:
            del state['onnxrt_']
        return state

    def __setstate__(self, state):
        """
        :epkg:`pickle` does not support functions.
        This method restores the function created when
        the runtime is `'numpy'`.
        """
        BaseEstimator.__setstate__(self, state)
        if 'numpy_code_' in state:
            model_onnx = load(io.BytesIO(state['onnx_']))
            opsets = onnx_model_opsets(model_onnx)
            self.onnxrt_ = self._build_onnx_runtime_numpy_compile(opsets)

    def fit(self, X, y=None, sample_weight=None, **kwargs):
        """
        Fits the estimator, converts to ONNX.

        :param X: features
        :param args: other arguments
        :param kwargs: fitting options
        """
        if not hasattr(self, 'estimator_'):
            self.estimator_ = clone(self.estimator)
        if y is None:
            if sample_weight is None:
                self.estimator_.fit(X, **kwargs)
            else:
                self.estimator_.fit(X, sample_weight=sample_weight, **kwargs)
        else:
            if sample_weight is None:
                self.estimator_.fit(X, y, **kwargs)
            else:
                self.estimator_.fit(
                    X, y, sample_weight=sample_weight, **kwargs)

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

    def onnx_parser(self):
        """
        Returns a parser for this model.
        """
        self._check_fitted_()
        if isinstance(self.onnxrt_, FunctionTransformer):
            def parser():
                # Types should be included as well.
                return [r[0] for r in self.onnx_io_names_['skl2onnx_outputs']]
            return parser
        return self.onnxrt_.onnx_parser()

    def onnx_shape_calculator(self):
        """
        Returns a shape calculator for this transform.
        """
        self._check_fitted_()

        if isinstance(self.onnxrt_, FunctionTransformer):
            def fct_shape_calculator(operator):
                # Types should be included as well.
                outputs = self.onnx_io_names_['skl2onnx_outputs']
                if len(operator.outputs) != len(outputs):
                    raise RuntimeError(  # pragma: no cover
                        "Mismatch between parser and shape calculator, "
                        "%r != %r." % (outputs, operator.outputs))
                for a, b in zip(operator.outputs, outputs):
                    a.type = b[1]
            return fct_shape_calculator

        calc = self.onnxrt_.onnx_shape_calculator()

        def shape_calculator(operator):
            return calc(operator)

        return shape_calculator

    def onnx_converter(self):
        """
        Returns a converter for this transform.
        """
        self._check_fitted_()

        if isinstance(self.onnxrt_, FunctionTransformer):

            def fct_converter(scope, operator, container):
                op = operator.raw_operator
                onnx_model = load(io.BytesIO(op.onnx_))
                add_onnx_graph(scope, operator, container, onnx_model)

            return fct_converter

        conv = self.onnxrt_.onnx_converter()

        def converter(scope, operator, container):
            op = operator.raw_operator
            onnx_model = op.onnxrt_.onnxrt_.obj
            conv(scope, operator, container, onnx_model=onnx_model)

        return converter


class OnnxSpeedupTransformer(TransformerMixin,
                             _OnnxPipelineStepSpeedup):
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
    :param nopython: used by :epkg:`numba` jitter

    Attributes created by method *fit*:

    * `estimator_`: cloned and trained version of *estimator*
    * `onnxrt_`: objet of type @see cl OnnxInference,
        :epkg:`sklearn:preprocessing:FunctionTransformer`
    * `numpy_code_`: python code equivalent to the inference
        method if the runtime is `'numpy'` or `'numba'`
    * `onnx_io_names_`: dictionary, additional information
        if the runtime is `'numpy'` or `'numba'`

    .. versionadded:: 0.7
    """

    def __init__(self, estimator, runtime='python', enforce_float32=True,
                 target_opset=None, conv_options=None, nopython=True):
        _OnnxPipelineStepSpeedup.__init__(
            self, estimator, runtime=runtime, enforce_float32=enforce_float32,
            target_opset=target_opset, conv_options=conv_options,
            nopython=nopython)

    def fit(self, X, y=None, sample_weight=None):  # pylint: disable=W0221
        """
        Trains based estimator.
        """
        if sample_weight is None:
            _OnnxPipelineStepSpeedup.fit(self, X, y)
        else:
            _OnnxPipelineStepSpeedup.fit(
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


class OnnxSpeedupRegressor(RegressorMixin,
                           _OnnxPipelineStepSpeedup):
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
    :param nopython: used by :epkg:`numba` jitter

    Attributes created by method *fit*:

    * `estimator_`: cloned and trained version of *estimator*
    * `onnxrt_`: objet of type @see cl OnnxInference,
        :epkg:`sklearn:preprocessing:FunctionTransformer`
    * `numpy_code_`: python code equivalent to the inference
        method if the runtime is `'numpy'` or `'numba'`
    * `onnx_io_names_`: dictionary, additional information
        if the runtime is `'numpy'` or `'numba'`

    .. versionadded:: 0.7
    """

    def __init__(self, estimator, runtime='python', enforce_float32=True,
                 target_opset=None, conv_options=None, nopython=True):
        _OnnxPipelineStepSpeedup.__init__(
            self, estimator, runtime=runtime, enforce_float32=enforce_float32,
            target_opset=target_opset, conv_options=conv_options,
            nopython=nopython)

    def fit(self, X, y, sample_weight=None):  # pylint: disable=W0221
        """
        Trains based estimator.
        """
        if sample_weight is None:
            _OnnxPipelineStepSpeedup.fit(self, X, y)
        else:
            _OnnxPipelineStepSpeedup.fit(
                self, X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """
        Transforms with *ONNX*.

        :param X: features
        :return: transformed features
        """
        return self.onnxrt_.transform(X)

    def raw_predict(self, X):
        """
        Transforms with *scikit-learn*.

        :param X: features
        :return: transformed features
        """
        return self.estimator_.predict(X)

    def assert_almost_equal(self, X, **kwargs):
        """
        Checks that ONNX and scikit-learn produces the same
        outputs.
        """
        expected = numpy.squeeze(self.raw_predict(X))
        got = numpy.squeeze(self.predict(X))
        assert_almost_equal(expected, got, **kwargs)


class OnnxSpeedupClassifier(ClassifierMixin,
                            _OnnxPipelineStepSpeedup):
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
    :param nopython: used by :epkg:`numba` jitter

    Attributes created by method *fit*:

    * `estimator_`: cloned and trained version of *estimator*
    * `onnxrt_`: objet of type @see cl OnnxInference,
        :epkg:`sklearn:preprocessing:FunctionTransformer`
    * `numpy_code_`: python code equivalent to the inference
        method if the runtime is `'numpy'` or `'numba'`
    * `onnx_io_names_`: dictionary, additional information
        if the runtime is `'numpy'` or `'numba'`

    .. versionadded:: 0.7
    """

    def __init__(self, estimator, runtime='python', enforce_float32=True,
                 target_opset=None, conv_options=None, nopython=True):
        if conv_options is None:
            conv_options = {'zipmap': False}
        _OnnxPipelineStepSpeedup.__init__(
            self, estimator, runtime=runtime, enforce_float32=enforce_float32,
            target_opset=target_opset, conv_options=conv_options,
            nopython=nopython)

    def fit(self, X, y, sample_weight=None):  # pylint: disable=W0221
        """
        Trains based estimator.
        """
        if sample_weight is None:
            _OnnxPipelineStepSpeedup.fit(self, X, y)
        else:
            _OnnxPipelineStepSpeedup.fit(
                self, X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """
        Transforms with *ONNX*.

        :param X: features
        :return: transformed features
        """
        pred = self.onnxrt_.transform(X)
        if isinstance(pred, tuple):
            return pred[0]
        return pred.iloc[:, 0].values

    def predict_proba(self, X):
        """
        Transforms with *ONNX*.

        :param X: features
        :return: transformed features
        """
        pred = self.onnxrt_.transform(X)
        if isinstance(pred, tuple):
            return pred[1]
        return pred.iloc[:, 1:].values

    def raw_predict(self, X):
        """
        Transforms with *scikit-learn*.

        :param X: features
        :return: transformed features
        """
        return self.estimator_.predict(X)

    def raw_predict_proba(self, X):
        """
        Transforms with *scikit-learn*.

        :param X: features
        :return: transformed features
        """
        return self.estimator_.predict_proba(X)

    def assert_almost_equal(self, X, **kwargs):
        """
        Checks that ONNX and scikit-learn produces the same
        outputs.
        """
        expected = numpy.squeeze(self.raw_predict_proba(X))
        got = numpy.squeeze(self.predict_proba(X))
        assert_almost_equal(expected, got, **kwargs)
        expected = numpy.squeeze(self.raw_predict(X))
        got = numpy.squeeze(self.predict(X))
        assert_almost_equal(expected, got, **kwargs)


class OnnxSpeedupCluster(ClusterMixin,
                         _OnnxPipelineStepSpeedup):
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
    :param nopython: used by :epkg:`numba` jitter

    Attributes created by method *fit*:

    * `estimator_`: cloned and trained version of *estimator*
    * `onnxrt_`: objet of type @see cl OnnxInference,
        :epkg:`sklearn:preprocessing:FunctionTransformer`
    * `numpy_code_`: python code equivalent to the inference
        method if the runtime is `'numpy'` or `'numba'`
    * `onnx_io_names_`: dictionary, additional information
        if the runtime is `'numpy'` or `'numba'`

    .. versionadded:: 0.7
    """

    def __init__(self, estimator, runtime='python', enforce_float32=True,
                 target_opset=None, conv_options=None, nopython=True):
        _OnnxPipelineStepSpeedup.__init__(
            self, estimator, runtime=runtime, enforce_float32=enforce_float32,
            target_opset=target_opset, conv_options=conv_options,
            nopython=nopython)

    def fit(self, X, y, sample_weight=None):  # pylint: disable=W0221
        """
        Trains based estimator.
        """
        if sample_weight is None:
            _OnnxPipelineStepSpeedup.fit(self, X, y)
        else:
            _OnnxPipelineStepSpeedup.fit(
                self, X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """
        Transforms with *ONNX*.

        :param X: features
        :return: transformed features
        """
        pred = self.onnxrt_.transform(X)
        if isinstance(pred, tuple):
            return pred[0]
        return pred.iloc[:, 0].values

    def transform(self, X):
        """
        Transforms with *ONNX*.

        :param X: features
        :return: transformed features
        """
        pred = self.onnxrt_.transform(X)
        if isinstance(pred, tuple):
            return pred[1]
        return pred.iloc[:, 1:].values

    def raw_predict(self, X):
        """
        Transforms with *scikit-learn*.

        :param X: features
        :return: transformed features
        """
        return self.estimator_.predict(X)

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
        expected = numpy.squeeze(self.raw_transform(X))
        got = numpy.squeeze(self.transform(X))
        assert_almost_equal(expected, got, **kwargs)
        expected = numpy.squeeze(self.raw_predict(X))
        got = numpy.squeeze(self.predict(X))
        assert_almost_equal(expected, got, **kwargs)
