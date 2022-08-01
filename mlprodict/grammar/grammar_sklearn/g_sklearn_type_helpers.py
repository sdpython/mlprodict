# -*- coding: utf-8 -*-
"""
@file
@brief Tiny helpers for scikit-learn exporters.
"""


def check_type(model, model_type):
    """
    Raises an exception if the model is not of the expected type.

    @param      model       *scikit-learn* model
    @param      model_type  expected type
    """
    if not isinstance(model, model_type):
        raise TypeError(  # pragma: no cover
            f"Model type {type(model)} is not of type {model_type}.")
