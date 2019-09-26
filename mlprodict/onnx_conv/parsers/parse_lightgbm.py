"""
@file
@brief Parsers for LightGBM booster.
"""
import numpy
from sklearn.base import ClassifierMixin
from skl2onnx._parse import _parse_sklearn_classifier, _parse_sklearn_simple_model


class WrappedLightGbmBooster:
    """
    A booster can be a classifier, a regressor.
    Trick to wrap it in a minimal function.
    """

    def __init__(self, booster):
        self.booster_ = booster
        self._model_dict = self.booster_.dump_model()
        self.classes_ = self._generate_classes(self._model_dict)
        self.n_features_ = len(self._model_dict['feature_names'])
        if self._model_dict['objective'].startswith('binary'):
            self.operator_name = 'LgbmClassifier'
        elif self._model_dict['objective'].startswith('regression'):
            self.operator_name = 'LgbmRegressor'
        else:
            raise NotImplementedError('Unsupported LightGbm objective: {}'.format(
                self._model_dict['objective']))
        if self._model_dict.get('average_output', False):
            self.boosting_type = 'rf'
        else:
            # Other than random forest, other boosting types do not affect later conversion.
            # Here `gbdt` is chosen for no reason.
            self.boosting_type = 'gbdt'

    def _generate_classes(self, model_dict):
        if model_dict['num_class'] == 1:
            return numpy.asarray([0, 1])
        return numpy.arange(model_dict['num_class'])


class WrappedLightGbmBoosterClassifier(ClassifierMixin):
    """
    Trick to wrap a LGBMClassifier into a class.
    """

    def __init__(self, wrapped):
        for k in {'boosting_type', '_model_dict', 'operator_name',
                  'classes_', 'booster_', 'n_features_'}:
            setattr(self, k, getattr(wrapped, k))


def lightgbm_parser(scope, model, inputs, custom_parsers=None):
    """
    Agnostic parser for LightGBM Booster.
    """
    if hasattr(model, "fit"):
        raise TypeError("This converter does not apply on type '{}'."
                        "".format(type(model)))

    wrapped = WrappedLightGbmBooster(model)
    if wrapped._model_dict['objective'].startswith('binary'):
        wrapped = WrappedLightGbmBoosterClassifier(wrapped)
        return _parse_sklearn_classifier(
            scope, wrapped, inputs, custom_parsers=custom_parsers)
    if wrapped._model_dict['objective'].startswith('regression'):
        return _parse_sklearn_simple_model(
            scope, wrapped, inputs, custom_parsers=custom_parsers)
    raise NotImplementedError(
        "Objective '{}' is not implemented yet.".format(
            wrapped._model_dict['objective']))
