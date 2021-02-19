"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
from skl2onnx.common._registration import (
    _converter_pool, _shape_calculator_pool)
try:
    from skl2onnx.common._registration import RegisteredConverter
except ImportError:  # pragma: no cover
    # sklearn-onnx <= 1.6.0
    RegisteredConverter = lambda fct, opts: fct
from .sklconv.tree_converters import (
    new_convert_sklearn_decision_tree_classifier,
    new_convert_sklearn_decision_tree_regressor,
    new_convert_sklearn_gradient_boosting_classifier,
    new_convert_sklearn_gradient_boosting_regressor,
    new_convert_sklearn_random_forest_classifier,
    new_convert_sklearn_random_forest_regressor)
from .sklconv.svm_converters import (
    new_convert_sklearn_svm_classifier,
    new_convert_sklearn_svm_regressor)
from .sklconv.function_transformer_converters import (
    new_calculate_sklearn_function_transformer_output_shapes,
    new_convert_sklearn_function_transformer)


_overwritten_operators = {
    #
    'SklearnOneClassSVM': RegisteredConverter(
        new_convert_sklearn_svm_regressor,
        _converter_pool['SklearnOneClassSVM'].get_allowed_options()),
    'SklearnSVR': RegisteredConverter(
        new_convert_sklearn_svm_regressor,
        _converter_pool['SklearnSVR'].get_allowed_options()),
    'SklearnSVC': RegisteredConverter(
        new_convert_sklearn_svm_classifier,
        _converter_pool['SklearnSVC'].get_allowed_options()),
    #
    'SklearnDecisionTreeRegressor': RegisteredConverter(
        new_convert_sklearn_decision_tree_regressor,
        _converter_pool['SklearnDecisionTreeRegressor'].get_allowed_options()),
    'SklearnDecisionTreeClassifier': RegisteredConverter(
        new_convert_sklearn_decision_tree_classifier,
        _converter_pool['SklearnDecisionTreeClassifier'].get_allowed_options()),
    #
    'SklearnExtraTreeRegressor': RegisteredConverter(
        new_convert_sklearn_decision_tree_regressor,
        _converter_pool['SklearnExtraTreeRegressor'].get_allowed_options()),
    'SklearnExtraTreeClassifier': RegisteredConverter(
        new_convert_sklearn_decision_tree_classifier,
        _converter_pool['SklearnExtraTreeClassifier'].get_allowed_options()),
    #
    'SklearnExtraTreesRegressor': RegisteredConverter(
        new_convert_sklearn_random_forest_regressor,
        _converter_pool['SklearnExtraTreesRegressor'].get_allowed_options()),
    'SklearnExtraTreesClassifier': RegisteredConverter(
        new_convert_sklearn_random_forest_classifier,
        _converter_pool['SklearnExtraTreesClassifier'].get_allowed_options()),
    #
    'SklearnFunctionTransformer': RegisteredConverter(
        new_convert_sklearn_function_transformer,
        _converter_pool['SklearnFunctionTransformer'].get_allowed_options()),
    #
    'SklearnGradientBoostingRegressor': RegisteredConverter(
        new_convert_sklearn_gradient_boosting_regressor,
        _converter_pool['SklearnGradientBoostingRegressor'].get_allowed_options()),
    'SklearnGradientBoostingClassifier': RegisteredConverter(
        new_convert_sklearn_gradient_boosting_classifier,
        _converter_pool['SklearnGradientBoostingClassifier'].get_allowed_options()),
    #
    'SklearnHistGradientBoostingRegressor': RegisteredConverter(
        new_convert_sklearn_random_forest_regressor,
        _converter_pool['SklearnHistGradientBoostingRegressor'].get_allowed_options()),
    'SklearnHistGradientBoostingClassifier': RegisteredConverter(
        new_convert_sklearn_random_forest_classifier,
        _converter_pool['SklearnHistGradientBoostingClassifier'].get_allowed_options()),
    #
    'SklearnRandomForestRegressor': RegisteredConverter(
        new_convert_sklearn_random_forest_regressor,
        _converter_pool['SklearnRandomForestRegressor'].get_allowed_options()),
    'SklearnRandomForestClassifier': RegisteredConverter(
        new_convert_sklearn_random_forest_classifier,
        _converter_pool['SklearnRandomForestClassifier'].get_allowed_options()),
}

_overwritten_shape_calculator = {
    "SklearnFunctionTransformer":
        new_calculate_sklearn_function_transformer_output_shapes,
}


def register_rewritten_operators(new_converters=None,
                                 new_shape_calculators=None):
    """
    Registers modified operators and returns the old values.

    :param new_converters: converters to rewrite or None
        to rewrite default ones
    :param new_shape_calculators: shape calculators to rewrite or
        None to rewrite default ones
    @return old converters, old shape calculators
    """
    old_conv = None
    old_shape = None

    if new_converters is None:
        for rew in _overwritten_operators:
            if rew not in _converter_pool:
                raise KeyError(  # pragma: no cover
                    "skl2onnx was not imported and '{}' was not registered."
                    "".format(rew))
        old_conv = {k: _converter_pool[k] for k in _overwritten_operators}
        _converter_pool.update(_overwritten_operators)
    else:
        for rew in new_converters:
            if rew not in _converter_pool:
                raise KeyError(  # pragma: no cover
                    "skl2onnx was not imported and '{}' was not registered."
                    "".format(rew))
        old_conv = {k: _converter_pool[k] for k in new_converters}
        _converter_pool.update(new_converters)

    if new_shape_calculators is None:
        for rew in _overwritten_shape_calculator:
            if rew not in _shape_calculator_pool:
                raise KeyError(  # pragma: no cover
                    "skl2onnx was not imported and '{}' was not registered."
                    "".format(rew))
        old_shape = {k: _shape_calculator_pool[k]
                     for k in _overwritten_shape_calculator}
        _shape_calculator_pool.update(_overwritten_shape_calculator)
    else:
        for rew in new_shape_calculators:
            if rew not in _shape_calculator_pool:
                raise KeyError(  # pragma: no cover
                    "skl2onnx was not imported and '{}' was not registered."
                    "".format(rew))
        old_shape = {k: _shape_calculator_pool[k]
                     for k in new_shape_calculators}
        _shape_calculator_pool.update(new_shape_calculators)

    return old_conv, old_shape
