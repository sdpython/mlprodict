# -*- encoding: utf-8 -*-
"""
@file
@brief Defines new operators.
"""


class OperatorSchema:
    """
    Defines a schema for operators added in this package
    such as @see cl TreeEnsembleRegressorDouble.
    """

    def __init__(self, name):
        self.name = name
        self.domain = 'mlprodict'
