"""
@file
@brief Shape calculator for LGBMClassifier, handles doubles.
"""
from skl2onnx.common.shape_calculator import (
    calculate_linear_regressor_output_shapes,
    calculate_linear_classifier_output_shapes
)


def calculate_lightgbm_output_shapes(operator):
    """
    Shape calculator for LightGBM Booster
    (see :epkg:`lightgbm`).
    """
    op = operator.raw_operator
    if not hasattr(op, "_model_dict"):
        raise TypeError("This converter does not apply on type '{}'."
                        "".format(type(op)))
    if op._model_dict['objective'].startswith('binary'):
        return calculate_linear_classifier_output_shapes(operator)
    if op._model_dict['objective'].startswith('regression'):
        return calculate_linear_regressor_output_shapes(operator)
    raise NotImplementedError(
        "Objective '{}' is not implemented yet.".format(
            op._model_dict['objective']))
