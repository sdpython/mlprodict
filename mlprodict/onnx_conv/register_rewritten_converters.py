"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
from skl2onnx.common._registration import _converter_pool

try:
    from skl2onnx.common._registration import RegisteredConverter
except ImportError:
    # sklearn-onnx <= 1.6.0
    RegisteredConverter = lambda fct, opts: fct

from .sklconv.ada_boost import convert_sklearn_ada_boost_regressor
from .sklconv.tree_converters import (
    convert_sklearn_decision_tree_regressor,
    convert_sklearn_gradient_boosting_regressor,
    convert_sklearn_random_forest_classifier,
    convert_sklearn_random_forest_regressor_converter,
)
from .sklconv.svm_converters import convert_sklearn_svm


_overwritten_operators = {
    'SklearnAdaBoostRegressor': RegisteredConverter(
        convert_sklearn_ada_boost_regressor,
        _converter_pool['SklearnAdaBoostRegressor'].get_allowed_options()),
    'SklearnDecisionTreeRegressor': RegisteredConverter(
        convert_sklearn_decision_tree_regressor,
        _converter_pool['SklearnDecisionTreeRegressor'].get_allowed_options()),
    'SklearnExtraTreesRegressor': RegisteredConverter(
        convert_sklearn_random_forest_regressor_converter,
        _converter_pool['SklearnExtraTreesRegressor'].get_allowed_options()),
    'SklearnGradientBoostingRegressor': RegisteredConverter(
        convert_sklearn_gradient_boosting_regressor,
        _converter_pool['SklearnGradientBoostingRegressor'].get_allowed_options()),
    'SklearnHistGradientBoostingClassifier': RegisteredConverter(
        convert_sklearn_random_forest_classifier,
        _converter_pool['SklearnHistGradientBoostingClassifier'].get_allowed_options()),
    'SklearnHistGradientBoostingRegressor': RegisteredConverter(
        convert_sklearn_random_forest_regressor_converter,
        _converter_pool['SklearnHistGradientBoostingRegressor'].get_allowed_options()),
    'SklearnOneClassSVM': RegisteredConverter(convert_sklearn_svm,
                                              _converter_pool['SklearnOneClassSVM'].get_allowed_options()),
    'SklearnRandomForestRegressor': RegisteredConverter(
        convert_sklearn_random_forest_regressor_converter,
        _converter_pool['SklearnRandomForestRegressor'].get_allowed_options()),
    'SklearnSVC': RegisteredConverter(
        convert_sklearn_svm,
        _converter_pool['SklearnSVC'].get_allowed_options()),
    'SklearnSVR': RegisteredConverter(
        convert_sklearn_svm,
        _converter_pool['SklearnSVR'].get_allowed_options()),
}


def register_rewritten_operators(new_values=None):
    """
    Registers modified operators and returns the old values.

    @param      new_values      operators to rewrite or None
                                to rewrite default ones
    @return                     old values
    """
    if new_values is None:
        for rew in _overwritten_operators:
            if rew not in _converter_pool:
                raise KeyError(
                    "skl2onnx was not imported and '{}' was not registered.".format(rew))
        old_values = {k: _converter_pool[k] for k in _overwritten_operators}
        _converter_pool.update(_overwritten_operators)
        return old_values
    else:
        for rew in new_values:
            if rew not in _converter_pool:
                raise KeyError(
                    "skl2onnx was not imported and '{}' was not registered.".format(rew))
        old_values = {k: _converter_pool[k] for k in new_values}
        _converter_pool.update(new_values)
        return old_values
