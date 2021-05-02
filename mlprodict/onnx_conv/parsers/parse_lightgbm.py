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
        self._model_dict = self.booster_.dump_model()
        self.classes_ = self._generate_classes(self._model_dict)
        self.n_features_ = len(self._model_dict['feature_names'])
        if self._model_dict['objective'].startswith('binary'):
            self.operator_name = 'LgbmClassifier'
        elif self._model_dict['objective'].startswith('regression'):  # pragma: no cover
            self.operator_name = 'LgbmRegressor'
        else:  # pragma: no cover
            raise NotImplementedError('Unsupported LightGbm objective: {}'.format(
                self._model_dict['objective']))
        if self._model_dict.get('average_output', False):
            self.boosting_type = 'rf'
        else:
            # Other than random forest, other boosting types do not affect later conversion.
            # Here `gbdt` is chosen for no reason.
            self.boosting_type = 'gbdt'

    @staticmethod
    def _generate_classes(model_dict):
        if model_dict['num_class'] == 1:
            return numpy.asarray([0, 1])
        return numpy.arange(model_dict['num_class'])


class WrappedLightGbmBoosterClassifier(ClassifierMixin):
    """
    Trick to wrap a LGBMClassifier into a class.
    """

    def __init__(self, wrapped):  # pylint: disable=W0231
        for k in {'boosting_type', '_model_dict', 'operator_name',
                  'classes_', 'booster_', 'n_features_'}:
            setattr(self, k, getattr(wrapped, k))


class MockWrappedLightGbmBoosterClassifier(WrappedLightGbmBoosterClassifier):
    """
    Mocked lightgbm.
    """

    def __init__(self, tree):  # pylint: disable=W0231
        self.dumped_ = tree

    def dump_model(self):
        "mock dump_model method"
        self.visited = True
        return self.dumped_


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
        if wrapped._model_dict['objective'].startswith('binary'):
            wrapped = WrappedLightGbmBoosterClassifier(wrapped)
            return _parse_sklearn_classifier(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        if wrapped._model_dict['objective'].startswith('regression'):  # pragma: no cover
            return _parse_sklearn_simple_model(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        raise NotImplementedError(  # pragma: no cover
            "Objective '{}' is not implemented yet.".format(
                wrapped._model_dict['objective']))

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
