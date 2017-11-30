# -*- coding: utf-8 -*-
"""
@file
@brief List of interpreted from scikit-learn model.
"""
import numpy
from .g_sklearn_type_helpers import check_type
from ..grammar.gactions import MLActionCst, MLActionVar, MLActionConcat, MLActionReturn
from ..grammar.gactions_num import MLActionAdd, MLActionSign
from ..grammar.gactions_tensor import MLActionTensorDot
from ..grammar.gmlactions import MLModel


def sklearn_logistic_regression(model, input_names=None, output_names=None, **kwargs):
    """
    Interprets a `logistic regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
    model into a *grammar* model (semantic graph representation).

    @param      model           *scikit-learn* model
    @param      input_names     name of the input features
    @param      output_names    name of the output predictions
    @return                     graph model

    If *input* is None or *output* is None, default values
    will be given to the outputs
    ``['Prediction', 'Score']`` for the outputs.
    If *input_names* is None, it wil be ``'Features'``.

    Additional parameters:
    - *with_loop*: False by default, *True* not implemented.
    """
    if kwargs.get('with_loop', False):
        raise NotImplementedError("Loop version is not implemented.")
    if output_names is None:
        output_names = ['Prediction', 'Score']
    if input_names is None:
        input_names = 'Features'

    from sklearn.linear_model import LogisticRegression
    check_type(model, LogisticRegression)
    if len(model.coef_.shape) > 1 and min(model.coef_.shape) != 1:
        raise NotImplementedError(
            "Multiclass is not implemented yet: coef_.shape={0}.".format(model.coef_.shape))
    coef = model.coef_.ravel()
    bias = numpy.float32(model.intercept_[0])

    gr_coef = MLActionCst(coef)
    gr_var = MLActionVar(coef, input_names)
    gr_bias = MLActionCst(bias)
    gr_dot = MLActionTensorDot(gr_coef, gr_var)
    gr_dist = MLActionAdd(gr_dot, gr_bias)
    gr_sign = MLActionSign(gr_dist)
    gr_conc = MLActionConcat(gr_sign, gr_dist)
    ret = MLActionReturn(gr_conc)
    return MLModel(ret, output_names, name=LogisticRegression.__name__)


def sklearn_linear_regression(model, input_names=None, output_names=None, **kwargs):
    """
    Converts a `linear regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_
    into a *grammar* model (semantic graph representation).

    @param      model           *scikit-learn* model
    @param      input_names     name of the input features
    @param      output_names    name of the output predictions
    @return                     graph model

    If *input* is None or *output* is None, default values
    will be given to the outputs
    ``['Prediction', 'Score']`` for the outputs.
    If *input_names* is None, it wil be ``'Features'``.

    Additional parameters:
    - *with_loop*: False by default, *True* not implemented.
    """
    if kwargs.get('with_loop', False):
        raise NotImplementedError("Loop version is not implemented.")
    if output_names is None:
        output_names = ['Prediction', 'Score']
    if input_names is None:
        input_names = 'Features'

    from sklearn.linear_model import LinearRegression
    check_type(model, LinearRegression)
    if len(model.coef_.shape) > 1 and min(model.coef_.shape) != 1:
        raise NotImplementedError(
            "MultiOutput is not implemented yet: coef_.shape={0}.".format(model.coef_.shape))
    coef = model.coef_.ravel()
    bias = numpy.float32(model.intercept_)

    gr_coef = MLActionCst(coef)
    gr_var = MLActionVar(coef, input_names)
    gr_bias = MLActionCst(bias)
    gr_dot = MLActionTensorDot(gr_coef, gr_var)
    gr_dist = MLActionAdd(gr_dot, gr_bias)
    ret = MLActionReturn(gr_dist)
    return MLModel(ret, output_names, name=LinearRegression.__name__)
