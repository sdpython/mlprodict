# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *onnx_conv*.
"""
import warnings
import numbers
import numpy
from skl2onnx._parse import _parse_sklearn_classifier
from .scorers import register_scorers


def _custom_parser_xgboost(scope, model, inputs, custom_parsers=None):
    """
    Custom parser for *XGBClassifier* and *LGBMClassifier*.
    """
    if custom_parsers is not None and model in custom_parsers:
        return custom_parsers[model](
            scope, model, inputs, custom_parsers=custom_parsers)
    if not all(isinstance(i, (numbers.Real, bool, numpy.bool_))
               for i in model.classes_):
        raise NotImplementedError(
            "Current converter does not support string labels.")
    return _parse_sklearn_classifier(scope, model, inputs)


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
        from .shape_calculators.shape_lightgbm import calculate_linear_classifier_output_shapes
        from .operator_converters.conv_lightgbm import convert_lightgbm
        try:
            update_registered_converter(
                LGBMClassifier, 'LgbmClassifier',
                calculate_linear_classifier_output_shapes,
                convert_lightgbm, parser=_parse_sklearn_classifier,
                options={'zipmap': [True, False], 'nocl': [True, False]})
        except TypeError:
            # skl2onnx <= 1.5
            update_registered_converter(
                LGBMClassifier, 'LgbmClassifier',
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

    try:
        from lightgbm import Booster
    except ImportError as e:  # pragma: no cover
        if exc:
            raise e
        else:
            warnings.warn(
                "Cannot register LGBMRegressor due to '{}'.".format(e))
            Booster = None
    if Booster is not None:
        from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
        from .operator_converters.conv_lightgbm import convert_lightgbm
        from .shape_calculators.shape_lightgbm import calculate_lightgbm_output_shapes
        from .parsers.parse_lightgbm import (
            lightgbm_parser, WrappedLightGbmBooster,
            WrappedLightGbmBoosterClassifier,
            shape_calculator_lightgbm_concat,
            converter_lightgbm_concat,
            MockWrappedLightGbmBoosterClassifier
        )
        update_registered_converter(
            Booster, 'LightGbmBooster', calculate_lightgbm_output_shapes,
            convert_lightgbm, parser=lightgbm_parser,
            options={'cast': [True, False]})
        update_registered_converter(
            WrappedLightGbmBooster, 'WrapperLightGbmBooster',
            calculate_lightgbm_output_shapes,
            convert_lightgbm, parser=lightgbm_parser)
        update_registered_converter(
            WrappedLightGbmBoosterClassifier, 'WrappedLightGbmBoosterClassifier',
            calculate_lightgbm_output_shapes,
            convert_lightgbm, parser=lightgbm_parser,
            options={'zipmap': [True, False], 'nocl': [True, False]})
        update_registered_converter(
            MockWrappedLightGbmBoosterClassifier, 'MockWrappedLightGbmBoosterClassifier',
            calculate_lightgbm_output_shapes,
            convert_lightgbm, parser=lightgbm_parser)
        update_registered_converter(
            None, 'LightGBMConcat',
            shape_calculator_lightgbm_concat,
            converter_lightgbm_concat)
        registered.append(Booster)
        registered.append(WrappedLightGbmBooster)
        registered.append(WrappedLightGbmBoosterClassifier)

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
            update_registered_converter(
                XGBClassifier, 'XGBoostXGBClassifier',
                calculate_linear_classifier_output_shapes,
                convert_xgboost, parser=_custom_parser_xgboost,
                options={'zipmap': [True, False], 'raw_scores': [True, False],
                         'nocl': [True, False]})
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
