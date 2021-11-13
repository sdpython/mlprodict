"""
@file
@brief Design to implement graph as parameter.

.. versionadded:: 0.8
"""


class OnnxGraphParameter:
    """
    Class wrapping a function to make it simple as
    a parameter.

    :param fct: function taking the list of inputs defined
        as @see cl OnnxVar, the function returns an @see cl OnnxVar
    :param inputs: list of input as @see cl OnnxVar
    """

    def __init__(self, fct, *inputs):
        self.fct = fct
        self.inputs = inputs

    def __repr__(self):
        "usual"
        return "%s(...)" % self.__class__.__name__


class if_then_else(OnnxGraphParameter):
    """
    Overloads class @see cl OnnxGraphParameter.
    """
    pass
