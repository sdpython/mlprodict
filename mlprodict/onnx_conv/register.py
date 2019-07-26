# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *onnx_conv*.
"""
import warnings


def register_converters(exc=True):
    """
    This functions registers additional converters
    to the list of converters :epkg:`sklearn-onnx` declares.

    @param      exc     if True, raises an exception if a converter cannot
                        registered (missing package for example)
    @return             list of models supported by the new converters
    """
    registered = []
    from skl2onnx import update_registered_converter
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        if exc:
            raise e
        else:
            warnings.warn(
                "Cannot register LGBMClassifier due to '{}'.".format(e))
            LGBMClassifier = None
    if LGBMClassifier is not None:
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from .operator_converters.conv_lightgbm import convert_lightgbm
        update_registered_converter(LGBMClassifier, 'LightGbmLGBMClassifier',
                                    calculate_linear_classifier_output_shapes,
                                    convert_lightgbm)
        registered.append(LGBMClassifier)

    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
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

    # end
    return registered
