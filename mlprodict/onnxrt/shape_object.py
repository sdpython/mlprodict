"""
@file
@brief Shape object.
"""


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

    def evaluated(self, **kwargs):
        """
        Evalutes the dimension.

        @param  kwargs      value for the variables.
        """
        if isinstance(self._dim, int):
            return self._dim
        if isinstance(self._dim, str):
            if len(kwargs) == 0:
                return self._dim
        raise NotImplementedError(
            "Not implemented for '{}'.".format(repr(self)))

    def _binaryop_(self, obj, strop, fct):
        """
        Applies binary operator to a dimension.

        @param      obj     other @see cl DimensionObject
        @param      strop   operator string
        @param      fct     python function which applies the operator
        @return             @see cl DimensionObject
        """
        o = DimensionObject._same_(obj)
        if isinstance(self._dim, int):
            if isinstance(o._dim, int):
                return DimensionObject(fct(self._dim, o._dim))
            if isinstance(o._dim, str):
                return DimensionObject("{}{}{}".format(self._dim, strop, o._dim))
            raise TypeError("Unable to handle type '{}'.".format(type(o._dim)))
        elif isinstance(self._dim, str):
            if isinstance(o._dim, int):
                return DimensionObject("{}{}{}".format(self._dim, strop, o._dim))
            elif isinstance(o._dim, str):
                return DimensionObject("({}){}({})".format(self._dim, strop, o._dim))
            raise TypeError("Unable to handle type '{}'.".format(type(o._dim)))
        else:
            raise TypeError(
                "Unable to handle type '{}'.".format(type(self._dim)))

    def __add__(self, obj):
        """
        usual
        """
        return self._binaryop_(obj, '+', lambda x, y: x + y)

    def __mul__(self, obj):
        """
        usual
        """
        return self._binaryop_(obj, '*', lambda x, y: x * y)


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
