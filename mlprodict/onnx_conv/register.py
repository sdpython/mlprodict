# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *onnx_conv*.
"""
import warnings
import numbers
import numpy
from skl2onnx._parse import _parse_sklearn_classifier
from skl2onnx.common.data_types import (
    SequenceType, DictionaryType, Int64TensorType, StringTensorType
)
from .scorers import register_scorers


def _custom_parser_xgboost(scope, model, inputs, custom_parsers=None):
    """
    Custom parser for *XGBClassifier*.
    """
    if custom_parsers is not None and model in custom_parsers:
        return custom_parsers[model](
            scope, model, inputs, custom_parsers=custom_parsers)
    if not all(isinstance(i, (numbers.Real, bool, numpy.bool_))
               for i in model.classes_):
        raise NotImplementedError(
            "Current converter does not support string labels.")
    return _parse_sklearn_classifier(scope, model, inputs)


def _custom_parser_lightgbm(scope, model, inputs, custom_parsers=None):
    """
    Custom parser for *LGBMClassifier*.
    """
    if custom_parsers is not None and model in custom_parsers:
        return custom_parsers[model](
            scope, model, inputs, custom_parsers=custom_parsers)
    if all(isinstance(i, (numbers.Real, bool, numpy.bool_))
           for i in model.classes_):
        label_type = Int64TensorType()
    else:
        label_type = StringTensorType()
    output_label = scope.declare_local_variable(
        'output_label', label_type)

    this_operator = scope.declare_local_operator(
        'LgbmClassifier', model)
    this_operator.inputs = inputs
    probability_map_variable = scope.declare_local_variable(
        'output_probability', SequenceType(DictionaryType(
            label_type, scope.tensor_type())))
    this_operator.outputs.append(output_label)
    this_operator.outputs.append(probability_map_variable)
    return this_operator.outputs


def _register_converters_lightgbm(exc=True):
    """
    This functions registers additional converters
    for :epkg:`lightgbm`.

    @param      exc     if True, raises an exception if a converter cannot
                        registered (missing package for example)
    @return             list of models supported by the new converters
    """
    registered = []
    from skl2onnx import update_registered_converter

    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:  # pragma: no cover
        if exc:
            raise e
        else:
            warnings.warn(
                "Cannot register LGBMClassifier due to '{}'.".format(e))
            LGBMClassifier = None
    if LGBMClassifier is not None:
        from .shape_calculators.conv_lightgbm import calculate_linear_classifier_output_shapes
        from .operator_converters.conv_lightgbm import convert_lightgbm
        try:
            update_registered_converter(LGBMClassifier, 'LgbmClassifier',
                                        calculate_linear_classifier_output_shapes,
                                        convert_lightgbm, parser=_custom_parser_lightgbm)
        except TypeError:
            # skl2onnx <= 1.5
            update_registered_converter(LGBMClassifier, 'LgbmClassifier',
                                        calculate_linear_classifier_output_shapes,
                                        convert_lightgbm)
        registered.append(LGBMClassifier)

    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:  # pragma: no cover
        if exc:
            raise e
        else:
            warnings.warn(
                "Cannot register LGBMRegressor due to '{}'.".format(e))
            LGBMRegressor = None
    if LGBMRegressor is not None:
        from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
        from .operator_converters.conv_lightgbm import convert_lightgbm
        update_registered_converter(LGBMRegressor, 'LightGbmLGBMRegressor',
                                    calculate_linear_regressor_output_shapes,
                                    convert_lightgbm)
        registered.append(LGBMRegressor)
    return registered


def _register_converters_xgboost(exc=True):
    """
    This functions registers additional converters
    for :epkg:`xgboost`.

    @param      exc     if True, raises an exception if a converter cannot
                        registered (missing package for example)
    @return             list of models supported by the new converters
    """
    registered = []
    from skl2onnx import update_registered_converter

    try:
        from xgboost import XGBClassifier
    except ImportError as e:  # pragma: no cover
        if exc:
            raise e
        else:
            warnings.warn(
                "Cannot register XGBClassifier due to '{}'.".format(e))
            XGBClassifier = None
    if XGBClassifier is not None:
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from .operator_converters.conv_xgboost import convert_xgboost
        try:
            update_registered_converter(XGBClassifier, 'XGBoostXGBClassifier',
                                        calculate_linear_classifier_output_shapes,
                                        convert_xgboost, parser=_custom_parser_xgboost)
        except TypeError:
            # skl2onnx <= 1.5
            update_registered_converter(XGBClassifier, 'XGBoostXGBClassifier',
                                        calculate_linear_classifier_output_shapes,
                                        convert_xgboost)
        registered.append(XGBClassifier)

    try:
        from xgboost import XGBRegressor
    except ImportError as e:  # pragma: no cover
        if exc:
            raise e
        else:
            warnings.warn(
                "Cannot register LGBMRegressor due to '{}'.".format(e))
            XGBRegressor = None
    if XGBRegressor is not None:
        from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
        from .operator_converters.conv_xgboost import convert_xgboost
        update_registered_converter(XGBRegressor, 'XGBoostXGBRegressor',
                                    calculate_linear_regressor_output_shapes,
                                    convert_xgboost)
        registered.append(XGBRegressor)
    return registered


def register_converters(exc=True):
    """
    This functions registers additional converters
    to the list of converters :epkg:`sklearn-onnx` declares.

    @param      exc     if True, raises an exception if a converter cannot
                        registered (missing package for example)
    @return             list of models supported by the new converters
    """
    ext = _register_converters_lightgbm(exc=exc)
    ext += _register_converters_xgboost(exc=exc)
    ext += register_scorers()
    return ext
