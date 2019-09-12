# -*- encoding: utf-8 -*-
"""
@file
@brief Overloads a conversion function.
"""
from collections import OrderedDict
import numpy
from sklearn.metrics.scorer import _PredictScorer
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType, DataType
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from skl2onnx.algebra.type_helper import guess_initial_types
from skl2onnx import convert_sklearn
from .rewritten_converters import register_rewritten_operators
from .register import register_converters
from .scorers import CustomScorerTransform


def convert_scorer(fct, initial_types, name=None,
                   target_opset=None, options=None,
                   dtype=numpy.float32,
                   custom_conversion_functions=None,
                   custom_shape_calculators=None,
                   custom_parsers=None):
    """
    Converts a scorer into :epkg:`ONNX` assuming
    there exists a converter associated to it.
    The function wraps the function into a custom
    transformer, then calls function *convert_sklearn*
    from :epkg:`sklearn-onnx`.

    @param  fct                         function to convert (or a scorer from
                                        :epkg:`scikit-learn`)
    @param  initial_types               types information
    @param  name                        name of the produced model
    @param  target_opset                to do it with a different target opset
    @param  options                     additional parameters for the conversion
    @param  dtype                       type to use to convert the model
    @param  custom_conversion_functions a dictionary for specifying the user customized
                                        conversion function, it takes precedence over
                                        registered converters
    @param  custom_shape_calculators    a dictionary for specifying the user
                                        customized shape calculator
                                        it takes precedence over registered
                                        shape calculators.
    @param  custom_parsers              parsers determine which outputs is expected
                                        for which particular task, default parsers are
                                        defined for classifiers, regressors, pipeline but
                                        they can be rewritten, *custom_parsers* is a dictionary
                                        ``{ type: fct_parser(scope, model, inputs,
                                        custom_parsers=None) }``
    @return                             :epkg:`ONNX` graph
    """
    if hasattr(fct, '_score_func'):
        kwargs = fct._kwargs
        fct = fct._score_func
    else:
        kwargs = None
    if name is None:
        name = "mlprodict_fct_ONNX(%s)" % fct.__name__
    tr = CustomScorerTransform(fct.__name__, fct, kwargs)
    return convert_sklearn(tr, initial_types=initial_types,
                           target_opset=target_opset, options=options,
                           dtype=dtype,
                           custom_conversion_functions=custom_conversion_functions,
                           custom_shape_calculators=custom_shape_calculators,
                           custom_parsers=custom_parsers)


def to_onnx(model, X=None, name=None, initial_types=None,
            target_opset=None, options=None,
            dtype=numpy.float32, rewrite_ops=False):
    """
    Converts a model using on :epkg:`sklearn-onnx`.

    @param      model           model to convert or a function
                                wrapped into :epkg:`_PredictScorer` with
                                function :epkg:`make_scorer`
    @param      X               training set (at least one row),
                                can be None, it is used to infered the
                                input types (*initial_types*)
    @param      initial_types   if *X* is None, then *initial_types* must be
                                defined
    @param      name            name of the produced model
    @param      target_opset    to do it with a different target opset
    @param      options         additional parameters for the conversion
    @param      dtype           type to use to convert the model
    @param      rewrite_ops     rewrites some existing converters,
                                the changes are permanent
    @return                     converted model

    The function rewrites function *to_onnx* from :epkg:`sklearn-onnx`
    but may changes a few converters if *rewrite_ops* is True.
    For example, :epkg:`ONNX` only supports *TreeEnsembleRegressor*
    for float but not for double. It becomes available
    if ``dtype=numpy.float64`` and ``rewrite_ops=True``.
    """
    if isinstance(model, OnnxOperatorMixin):
        if options is not None:
            raise NotImplementedError(
                "options not yet implemented for OnnxOperatorMixin.")
        return model.to_onnx(X=X, name=name, dtype=dtype,
                             target_opset=target_opset)
    if rewrite_ops:
        old_values = register_rewritten_operators()
        register_converters()
    else:
        old_values = None

    def _guess_type_(X, itype, dtype):
        initial_types = guess_initial_types(X, itype)
        if dtype is None:
            raise RuntimeError("dtype cannot be None")
        if isinstance(dtype, FloatTensorType):
            dtype = numpy.float32
        elif isinstance(dtype, DoubleTensorType):
            dtype = numpy.float64
        new_dtype = dtype
        if isinstance(dtype, numpy.ndarray):
            new_dtype = dtype.dtype
        elif isinstance(dtype, DataType):
            new_dtype = numpy.float32
        if new_dtype not in (numpy.float32, numpy.float64, numpy.int64,
                             numpy.int32):
            raise NotImplementedError(
                "dtype should be real not {} ({})".format(new_dtype, dtype))
        return initial_types, dtype, new_dtype

    if isinstance(model, _PredictScorer):
        if X is not None and not isinstance(X, OrderedDict):
            raise ValueError("For a scorer, parameter X should be a OrderedDict not {}."
                             "".format(type(X)))
        if initial_types is None:
            dts = []
            initial_types = []
            for k, v in X.items():
                it, _, ndt = _guess_type_(v, None, dtype)
                for i in range(len(it)):  # pylint: disable=C0200
                    it[i] = (k, it[i][1])  # pylint: disable=C0200
                initial_types.extend(it)
                dts.append(ndt)
            ndt = set(dts)
            if len(ndt) != 1:
                raise RuntimeError(
                    "Multiple dtype is not efficient {}.".format(ndt))
            dtype = dts[0]
            new_dtype = dts[0]
        res = convert_scorer(model, initial_types, name=name,
                             target_opset=target_opset, options=options,
                             dtype=new_dtype)
    else:
        if name is None:
            name = "mlprodict_ONNX(%s)" % model.__class__.__name__

        initial_types, dtype, new_dtype = _guess_type_(X, initial_types, dtype)
        try:
            res = convert_sklearn(model, initial_types=initial_types, name=name,
                                  target_opset=target_opset, options=options,
                                  dtype=new_dtype)
        except TypeError:
            # older version of sklearn-onnx
            res = convert_sklearn(model, initial_types=initial_types, name=name,
                                  target_opset=target_opset, options=options)

    if old_values is not None:
        register_rewritten_operators(old_values)
    return res
