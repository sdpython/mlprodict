"""
@file
@brief Parsers for LightGBM booster.
"""
import numpy
from sklearn.base import ClassifierMixin
from skl2onnx._parse import _parse_sklearn_classifier, _parse_sklearn_simple_model
from skl2onnx.common._apply_operation import apply_concat, apply_cast
from skl2onnx.common.data_types import guess_proto_type
from skl2onnx.proto import onnx_proto


class WrappedLightGbmBooster:
    """
    A booster can be a classifier, a regressor.
    Trick to wrap it in a minimal function.
    """

    def __init__(self, booster):
        self.booster_ = booster
        self.n_features_ = self.booster_.feature_name()
        self.objective_ = self.get_objective()
        if self.objective_.startswith('binary'):
            self.operator_name = 'LgbmClassifier'
            self.classes_ = self._generate_classes(booster)
        elif self.objective_.startswith('multiclass'):
            self.operator_name = 'LgbmClassifier'
            self.classes_ = self._generate_classes(booster)
        elif self.objective_.startswith('regression'):  # pragma: no cover
            self.operator_name = 'LgbmRegressor'
        else:  # pragma: no cover
            raise NotImplementedError(
                'Unsupported LightGbm objective: %r.' % self.objective_)
        average_output = self.booster_.attr('average_output')
        if average_output:
            self.boosting_type = 'rf'
        else:
            # Other than random forest, other boosting types do not affect later conversion.
            # Here `gbdt` is chosen for no reason.
            self.boosting_type = 'gbdt'

    @staticmethod
    def _generate_classes(booster):
        if isinstance(booster, dict):
            num_class = booster['num_class']
        else:
            num_class = booster.attr('num_class')
        if num_class is None:
            dp = booster.dump_model(num_iteration=1)
            num_class = dp['num_class']
        if num_class == 1:
            return numpy.asarray([0, 1])
        return numpy.arange(num_class)

    def get_objective(self):
        "Returns the objective."
        if hasattr(self, 'objective_') and self.objective_ is not None:
            return self.objective_
        objective = self.booster_.attr('objective')
        if objective is not None:
            return objective
        dp = self.booster_.dump_model(num_iteration=1)
        return dp['objective']


class WrappedLightGbmBoosterClassifier(ClassifierMixin):
    """
    Trick to wrap a LGBMClassifier into a class.
    """

    def __init__(self, wrapped):  # pylint: disable=W0231
        for k in {'boosting_type', '_model_dict', '_model_dict_info',
                  'operator_name', 'classes_', 'booster_', 'n_features_',
                  'objective_', 'boosting_type', 'n_features_'}:
            if hasattr(wrapped, k):
                setattr(self, k, getattr(wrapped, k))


class MockWrappedLightGbmBoosterClassifier(WrappedLightGbmBoosterClassifier):
    """
    Mocked lightgbm.
    """

    def __init__(self, tree):  # pylint: disable=W0231
        self.dumped_ = tree
        self.is_mock = True

    def dump_model(self):
        "mock dump_model method"
        self.visited = True
        return self.dumped_

    def feature_name(self):
        "Returns binary features names."
        return [0, 1]

    def attr(self, key):
        "Returns default values for common attributes."
        if key == 'objective':
            return "binary"
        if key == 'num_class':
            return 1
        if key == 'average_output':
            return None
        raise KeyError(  # pragma: no cover
            "No response for %r." % key)


def lightgbm_parser(scope, model, inputs, custom_parsers=None):
    """
    Agnostic parser for LightGBM Booster.
    """
    if hasattr(model, "fit"):
        raise TypeError(  # pragma: no cover
            "This converter does not apply on type '{}'."
            "".format(type(model)))

    if len(inputs) == 1:
        wrapped = WrappedLightGbmBooster(model)
        objective = wrapped.get_objective()
        if objective.startswith('binary'):
            wrapped = WrappedLightGbmBoosterClassifier(wrapped)
            return _parse_sklearn_classifier(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        if objective.startswith('multiclass'):
            wrapped = WrappedLightGbmBoosterClassifier(wrapped)
            return _parse_sklearn_classifier(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        if objective.startswith('regression'):  # pragma: no cover
            return _parse_sklearn_simple_model(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        raise NotImplementedError(  # pragma: no cover
            "Objective '{}' is not implemented yet.".format(objective))

    # Multiple columns
    this_operator = scope.declare_local_operator('LightGBMConcat')
    this_operator.raw_operator = model
    this_operator.inputs = inputs
    var = scope.declare_local_variable(
        'Xlgbm', inputs[0].type.__class__([None, None]))
    this_operator.outputs.append(var)
    return lightgbm_parser(scope, model, this_operator.outputs,
                           custom_parsers=custom_parsers)


def shape_calculator_lightgbm_concat(operator):
    """
    Shape calculator for operator *LightGBMConcat*.
    """
    pass


def converter_lightgbm_concat(scope, operator, container):
    """
    Converter for operator *LightGBMConcat*.
    """
    op = operator.raw_operator
    options = container.get_options(op, dict(cast=False))
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:  # pylint: disable=E1101
        proto_dtype = onnx_proto.TensorProto.FLOAT  # pylint: disable=E1101
    if options['cast']:
        concat_name = scope.get_unique_variable_name('cast_lgbm')
        apply_cast(scope, concat_name, operator.outputs[0].full_name, container,
                   operator_name=scope.get_unique_operator_name('cast_lgmb'),
                   to=proto_dtype)
    else:
        concat_name = operator.outputs[0].full_name

    apply_concat(scope, [_.full_name for _ in operator.inputs],
                 concat_name, container, axis=1)
