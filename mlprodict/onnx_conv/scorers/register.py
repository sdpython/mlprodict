# -*- encoding: utf-8 -*-
"""
@file
@brief Registers new converters.
"""
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from skl2onnx import (
    update_registered_converter,
    update_registered_parser)
from skl2onnx.common.data_types import guess_tensor_type
from skl2onnx.common._apply_operation import apply_identity


class CustomScorerTransform(BaseEstimator, TransformerMixin):
    """
    Wraps a scoring function into a transformer. Function @see fn
    register_scorers must be called to register the converter
    associated to this transform. It takes two inputs, expected values
    and predicted values and returns a score for each observation.
    """

    def __init__(self, name, fct, kwargs):
        """
        @param      name        function name
        @param      fct         python function
        @param      kwargs      parameters function
        """
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.name_fct = name
        self._fct = fct
        self.kwargs = kwargs

    def __repr__(self):  # pylint: disable=W0222
        return "{}('{}', {}, {})".format(
            self.__class__.__name__, self.name_fct,
            self._fct.__name__, self.kwargs)


def custom_scorer_transform_parser(scope, model, inputs, custom_parsers=None):
    """
    This function updates the inputs and the outputs for
    a @see cl CustomScorerTransform.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., *OneHotEncoder*
        or *LogisticRegression*)
    :param inputs: A list of variables
    :param custom_parsers: parsers determines which outputs is expected
        for which particular task, default parsers are defined for
        classifiers, regressors, pipeline but they can be rewritten,
        *custom_parsers* is a dictionary
        ``{ type: fct_parser(scope, model, inputs, custom_parsers=None) }``
    :return: A list of output variables which will be passed to next
        stage
    """
    if custom_parsers is not None:  # pragma: no cover
        raise NotImplementedError(
            "Case custom_parsers not empty is not implemented yet.")
    if isinstance(model, str):
        raise RuntimeError(  # pragma: no cover
            "Parameter model must be an object not a "
            "string '{0}'.".format(model))
    if len(inputs) != 2:
        raise RuntimeError(  # pragma: no cover
            "Two inputs expected not {}.".format(len(inputs)))
    alias = 'Mlprodict' + model.__class__.__name__
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs

    scores = scope.declare_local_variable(
        'scores', guess_tensor_type(inputs[0].type))
    this_operator.outputs.append(scores)
    return this_operator.outputs


def custom_scorer_transform_shape_calculator(operator):
    """
    Computes the output shapes for a @see cl CustomScorerTransform.
    """
    if len(operator.inputs) != 2:
        raise RuntimeError("Two inputs expected.")  # pragma: no cover
    if len(operator.outputs) != 1:
        raise RuntimeError("One output expected.")  # pragma: no cover

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, 1]


def custom_scorer_transform_converter(scope, operator, container):
    """
    Selects the appropriate converter for a @see cl CustomScorerTransform.
    """
    op = operator.raw_operator
    name = op.name_fct
    this_operator = scope.declare_local_operator('fct_' + name)
    this_operator.raw_operator = op
    this_operator.inputs = operator.inputs

    score_name = scope.declare_local_variable(
        'scores', operator.inputs[0].type)
    this_operator.outputs.append(score_name)
    apply_identity(scope, score_name.full_name,
                   operator.outputs[0].full_name, container)


def register_scorers():
    """
    Registers operators for @see cl CustomScorerTransform.
    """
    from .cdist_score import score_cdist_sum, convert_score_cdist_sum
    done = []
    update_registered_parser(
        CustomScorerTransform,
        custom_scorer_transform_parser)

    update_registered_converter(
        CustomScorerTransform,
        'MlprodictCustomScorerTransform',
        custom_scorer_transform_shape_calculator,
        custom_scorer_transform_converter)
    done.append(CustomScorerTransform)

    update_registered_converter(
        score_cdist_sum, 'fct_score_cdist_sum',
        None, convert_score_cdist_sum,
        options={'cdist': [None, 'single-node']})

    return done
