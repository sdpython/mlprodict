"""
@file
@brief Shape object.
"""


class ShapeOperator:
    """
    Base class for all shapes operator.
    """

    def __init__(self, name, fct, fct_string, *args):
        """
        @param      name        display name of the operator
        @param      fct         function doing the operator
                                if argument are numeric
        @param      fct_string  function represented as a string
        @param      args        argument of the operator
        """
        self._name = name
        self._fct = fct
        self._fct_string = fct_string
        self._args = args
        for a in self._args:
            if not isinstance(a, DimensionObject):
                raise TypeError(
                    "All arguments must be of type DimensionObject not '{}'.".format(type(a)))

    def __repr__(self):
        """
        usual
        """
        return "{0}('{1}', {2}, '{2}', {3})".format(
            self.__class__.__name__, self._name,
            self._fct_string, self._args)

    def to_string(self):
        """
        Displays as a string.

        @return     a string
        """
        raise NotImplementedError(
            "Operator '{}' does not implement 'to_string': {}.".format(
                self.__class__.__name__, repr(self)))

    def evaluate(self, **kwargs):
        """
        Evalutes the operator.

        @param  kwargs      value for the variables.
        @return             string or integer
        """
        args = []
        for a in self._args:
            a = DimensionObject._same_(a)
            v = a.evaluate(**kwargs)
            args.append(v)
        try:
            res = self._fct(*args)
        except TypeError as e:
            raise RuntimeError(
                "Unable to evaluate operator {} due to {}".format(repr(self), e))
        return res


class ShapeBinaryOperator(ShapeOperator):
    """
    Base class for shape binary operator.
    """

    def __init__(self, name, fct, fct_string, x, y):
        """
        @param      name        display name of the operator
        @param      fct         function doing the operator
                                if argument are numeric
        @param      fct_string  function represented as a string
        @param      x           first argument
        @param      y           second argument
        """
        ShapeOperator.__init__(self, name, fct, fct_string, x, y)
        if isinstance(x, tuple):
            raise TypeError('x cannot be a tuple')
        if isinstance(y, tuple):
            raise TypeError('y cannot be a tuple')

    def to_string(self):
        """
        Applies binary operator to a dimension.

        @return             a string
        """
        x, y = self._args  # pylint: disable=W0632
        if isinstance(x._dim, int):
            if isinstance(y._dim, int):
                return DimensionObject(self._fct(x._dim, y._dim)).to_string()
            if isinstance(y._dim, str):
                return DimensionObject("{}{}{}".format(x._dim, self._name, y._dim)).to_string()
            raise TypeError("Unable to handle type '{}'.".format(type(y._dim)))
        elif isinstance(x._dim, str):
            if isinstance(y._dim, int):
                return DimensionObject("{}{}{}".format(x._dim, self._name, y._dim)).to_string()
            elif isinstance(y._dim, str):
                return DimensionObject("({}){}({})".format(x._dim, self._name, y._dim)).to_string()
            raise TypeError("Unable to handle type '{}'.".format(type(y._dim)))
        else:
            raise TypeError(
                "Unable to handle type '{}'.".format(type(x._dim)))


class ShapeOperatorAdd(ShapeBinaryOperator):
    """
    Shape addition.
    """

    def __init__(self, x, y):
        ShapeBinaryOperator.__init__(
            self, '+', lambda a, b: a + b, 'lambda a, b: a + b', x, y)

    def __repr__(self):
        """
        Displays a string.

        @return             a string
        """
        return "{0}({1}, {2})".format(
            self.__class__.__name__, repr(self._args[0]), repr(self._args[1]))


class ShapeOperatorMul(ShapeBinaryOperator):
    """
    Shape multiplication.
    """

    def __init__(self, x, y):
        ShapeBinaryOperator.__init__(
            self, '*', lambda a, b: a * b, 'lambda a, b: a * b', x, y)

    def __repr__(self):
        """
        Displays a string.

        @return             a string
        """
        return "{0}({1}, {2})".format(
            self.__class__.__name__, repr(self._args[0]), repr(self._args[1]))


class DimensionObject:
    """
    One dimension of a shape.
    """

    def __init__(self, obj):
        """
        @param  obj     int or @see cl DimensionObject or None to
                        specify something unknown
        """
        if obj is None:
            self._dim = 'x'
        else:
            self._dim = obj
        if isinstance(self._dim, tuple):
            raise TypeError("obj cannot be a tuple")

    @property
    def dim(self):
        """
        Returns the dimension.
        """
        return self._dim

    def __repr__(self):
        """
        usual
        """
        if isinstance(self._dim, int):
            return "DimensionObject({})".format(self._dim)
        if isinstance(self._dim, DimensionObject):
            return repr(self._dim)
        if isinstance(self._dim, ShapeOperator):
            return "DimensionObject({})".format(repr(self._dim))
        return "DimensionObject('{}')".format(self._dim)

    @staticmethod
    def _same_(obj):
        """
        Returns *obj* if *obj* is @see cl DimensionObject
        otherwise converts it.
        """
        if isinstance(obj, DimensionObject):
            return obj
        return DimensionObject(obj)

    def to_string(self):
        """
        Represents the dimension as a string.
        """
        if isinstance(self._dim, int):
            return '{}'.format(self._dim)
        if isinstance(self._dim, ShapeOperator):
            return self._dim.to_string()
        if isinstance(self._dim, str):
            return self._dim
        raise NotImplementedError(
            "Not implemented for '{}'.".format(repr(self)))

    def evaluate(self, **kwargs):
        """
        Evalutes the dimension.

        @param  kwargs      value for the variables.
        @return             string or integer
        """
        if isinstance(self._dim, int):
            return self._dim
        if isinstance(self._dim, (ShapeOperator, DimensionObject)):
            return self._dim.evaluate(**kwargs)
        if isinstance(self._dim, str) and self._dim in kwargs:
            return kwargs[self._dim]
        raise NotImplementedError(
            "Not implemented for '{}'.".format(repr(self)))

    def __add__(self, obj):
        """
        usual
        """
        return DimensionObject(
            ShapeOperatorAdd(self, DimensionObject._same_(obj)))

    def __mul__(self, obj):
        """
        usual
        """
        return DimensionObject(
            ShapeOperatorMul(self, DimensionObject._same_(obj)))


class ShapeObject:
    """
    Handles mathematical operations around shapes.
    """

    def __init__(self, shape):
        """

        """
        self._shape = list(shape)

    @property
    def shape(self):
        """
        Returns the stored shape.
        """
        return tuple(self._shape)
