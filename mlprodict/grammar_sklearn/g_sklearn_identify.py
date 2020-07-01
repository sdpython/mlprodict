# -*- coding: utf-8 -*-
"""
@file
@brief Helpers to identify an interpreter.
"""
from ..tools import change_style
from .g_sklearn_linear_model import sklearn_logistic_regression, sklearn_linear_regression
from .g_sklearn_preprocessing import sklearn_standard_scaler
from .g_sklearn_tree import sklearn_decision_tree_regressor


def __pep8():  # pragma: no cover
    assert sklearn_decision_tree_regressor
    assert sklearn_linear_regression
    assert sklearn_logistic_regression
    assert sklearn_standard_scaler


def identify_interpreter(model):
    """
    Identifies the interpreter for a *scikit-learn* model.

    @param      model       model to identify
    @return                 interpreter
    """
    class_name = model.__class__.__name__
    pyname = change_style(class_name)
    skconv = "sklearn_" + pyname
    loc = globals().copy()
    convs = {k: v for k, v in loc.items() if k.startswith("sklearn")}
    if len(convs) == 0:
        raise ValueError(  # pragma: no cover
            "No found interpreters, possibilities=\n{0}".format(
                "\n".join(sorted(loc.keys()))))
    if skconv in convs:
        return convs[skconv]
    raise NotImplementedError(  # pragma: no cover
        "Model class '{0}' is not yet implemented. Available interpreters:\n{1}".format(
            class_name, "\n".join(
                sorted(
                    convs.keys()))))
