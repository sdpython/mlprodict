"""
@file
@brief Action definition.
"""
from .gtypes import MLNumTypeFloat32, MLNumTypeFloat64, MLNumTypeBool
from .gactions import MLActionBinary, MLActionFunctionCall


class MLActionAdd(MLActionBinary):
    """
    Addition
    """

    def __init__(self, act1, act2):
        """
        @param  act1    first element
        @param  act2    second element
        """
        MLActionBinary.__init__(self, act1, act2, "+")
        if type(act1.output) != type(act2.output):
            raise TypeError(  # pragma: no cover
                "Not the same input type {0} != {1}".format(
                    type(act1.output), type(act2.output)))

    def execute(self, **kwargs):
        MLActionBinary.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(res[0] + res[1])


class MLActionSign(MLActionFunctionCall):
    """
    Sign of an expression: 1=positive, 0=negative.
    """

    def __init__(self, act1):
        """
        @param  act1    first element
        """
        MLActionFunctionCall.__init__(self, "sign", act1.output, act1)
        if not isinstance(act1.output, (MLNumTypeFloat32, MLNumTypeFloat64)):
            raise TypeError(  # pragma: no cover
                "The input action must produce float32 or float64 not '{0}'".format(type(act1.output)))

    def execute(self, **kwargs):
        MLActionFunctionCall.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(self.output.softcast(1 if res[0] >= 0 else 0))


class MLActionTestInf(MLActionBinary):
    """
    Operator ``<``.
    """

    def __init__(self, act1, act2):
        """
        @param  act1    first element
        @param  act2    second element
        """
        MLActionBinary.__init__(self, act1, act2, "<=")
        if type(act1.output) != type(act2.output):
            raise TypeError(  # pragma: no cover
                "Not the same input type {0} != {1}".format(
                    type(act1.output), type(act2.output)))
        self.output = MLNumTypeBool()

    def execute(self, **kwargs):
        MLActionBinary.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(self.output.softcast(res[0] <= res[1]))


class MLActionTestEqual(MLActionBinary):
    """
    Operator ``==``.
    """

    def __init__(self, act1, act2):
        """
        @param  act1    first element
        @param  act2    second element
        """
        MLActionBinary.__init__(self, act1, act2, "==")
        if type(act1.output) != type(act2.output):
            raise TypeError(  # pragma: no cover
                "Not the same input type {0} != {1}".format(
                    type(act1.output), type(act2.output)))
        self.output = MLNumTypeBool()

    def execute(self, **kwargs):
        MLActionBinary.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(self.output.softcast(res[0] == res[1]))
