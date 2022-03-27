"""
@file
@brief Action definition.
"""
import numpy
from .gactions import MLActionFunctionCall


class MLActionTensorDot(MLActionFunctionCall):
    """
    Scalar product.
    """

    def __init__(self, act1, act2):
        """
        @param  act1    first tensor
        @param  act2    second tensor
        """
        MLActionFunctionCall.__init__(self, "adot", act1.output, act1, act2)
        # dot product takes two vectors and returns a float
        self.output = act1.output.element_type

    def _optional_parameters(self):
        return str(self.inputs[0].dim[0])

    def execute(self, **kwargs):
        """
        Addition
        """
        MLActionFunctionCall.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(self.output.softcast(numpy.dot(res[0], res[1])))


class MLActionTensorTake(MLActionFunctionCall):
    """
    Extracts an element of the tensor.
    """

    def __init__(self, tens, ind):
        """
        @param  tens    tensor
        @param  ind     index
        """
        MLActionFunctionCall.__init__(self, "atake", tens.output, tens, ind)
        self.output = tens.output.element_type

    def _optional_parameters(self):
        return str(self.inputs[0].dim[0])

    def execute(self, **kwargs):
        """
        Addition
        """
        MLActionFunctionCall.execute(self, **kwargs)
        res = self.ChildrenResults
        if res[1] < 0:
            raise ValueError(  # pragma: no cover
                "Cannot take element {0}".format(res[1]))
        if res[1] >= len(res[0]):
            raise ValueError(  # pragma: no cover
                "Cannot take element {0} >= size={1}".format(res[1], len(res[0])))
        return self.output.validate(self.output.softcast(res[0][res[1]]))


class MLActionTensorVector(MLActionFunctionCall):
    """
    Tensor operation.
    """

    def __init__(self, act1, act2, name, fct):
        """
        @param  act1    first tensor
        @param  act2    second tensor
        @param  name    operator name
        @param  fct     function
        """
        MLActionFunctionCall.__init__(self, name, act1.output, act1, act2)
        self.output = act1.output
        self.fct = fct

    def _optional_parameters(self):
        return str(self.inputs[0].dim[0])

    def execute(self, **kwargs):
        """
        Addition
        """
        MLActionFunctionCall.execute(self, **kwargs)
        res = self.ChildrenResults
        return self.output.validate(self.fct(res[0], res[1]))


class MLActionTensorSub(MLActionTensorVector):
    """
    Tensor soustraction.
    """

    def __init__(self, act1, act2):
        """
        @param  act1    first tensor
        @param  act2    second tensor
        """
        MLActionTensorVector.__init__(
            self, act1, act2, "asub", lambda v1, v2: v1 - v2)


class MLActionTensorMul(MLActionTensorVector):
    """
    Tensor multiplication.
    """

    def __init__(self, act1, act2):
        """
        @param  act1    first tensor
        @param  act2    second tensor
        """
        MLActionTensorVector.__init__(  # pragma: no cover
            self, act1, act2, "amul", lambda v1, v2: numpy.multiply(v1, v2))


class MLActionTensorDiv(MLActionTensorVector):
    """
    Tensor division.
    """

    def __init__(self, act1, act2):
        """
        @param  act1    first tensor
        @param  act2    second tensor
        """
        MLActionTensorVector.__init__(
            self, act1, act2, "adiv", lambda v1, v2: numpy.divide(v1, v2))


class MLActionTensorAdd(MLActionTensorVector):
    """
    Tensor addition.
    """

    def __init__(self, act1, act2):
        """
        @param  act1    first tensor
        @param  act2    second tensor
        """
        MLActionTensorVector.__init__(  # pragma: no cover
            self, act1, act2, "aadd", lambda v1, v2: v1 + v2)
