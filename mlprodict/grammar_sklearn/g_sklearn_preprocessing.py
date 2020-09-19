# -*- coding: utf-8 -*-
"""
@file
@brief Converters from scikit-learn model.
"""
import numpy
from .g_sklearn_type_helpers import check_type
from .grammar.gactions import MLActionVar, MLActionCst, MLActionReturn
from .grammar.gactions_tensor import MLActionTensorDiv, MLActionTensorSub
from .grammar.gmlactions import MLModel


def sklearn_standard_scaler(model, input_names=None, output_names=None, **kwargs):
    """
    Converts a `standard scaler <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_
    model into a *grammar* model (semantic graph representation).

    @param      model           scikit-learn model
    @param      input_names     name of the input features
    @param      output_names    name of the output predictions
    @param      kwargs          additional parameters (none)
    @return                     graph model

    If *input* is None or *output* is None, default values
    will be given to the outputs
    ``['Prediction', 'Score']`` for the outputs.
    If *input_names* is None, it wil be ``'Features'``.

    No additional parameters is considered.
    """
    if output_names is None:
        output_names = ['Prediction', 'Score']
    if input_names is None:
        input_names = 'Features'

    from sklearn.preprocessing import StandardScaler
    check_type(model, StandardScaler)

    lmean = MLActionCst(model.mean_.ravel().astype(numpy.float32))
    lscale = MLActionCst(model.scale_.ravel().astype(numpy.float32))

    lvar = MLActionVar(model.var_.astype(numpy.float32), input_names)
    lno = MLActionTensorSub(lvar, lmean)
    lno = MLActionTensorDiv(lno, lscale)
    ret = MLActionReturn(lno)
    return MLModel(ret, output_names, name=StandardScaler.__name__)
