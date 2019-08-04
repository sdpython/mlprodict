"""
@file
@brief Shape object.
"""
import numpy


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
        if obj is None or obj == 0:
            self._dim = 'x'
        elif isinstance(obj, (int, str, ShapeOperator, DimensionObject)):
            self._dim = obj
        else:
            raise TypeError("Unexpected type for obj: {}".format(type(obj)))

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
        if isinstance(self._dim, str):
            return self
        raise NotImplementedError(
            "Not implemented for '{}'.".format(repr(self)))

    def __eq__(self, v):
        """
        usual
        """
        if isinstance(v, (int, str)):
            return self._dim == v
        if isinstance(v, DimensionObject):
            return v == self._dim
        if isinstance(v, ShapeOperator):
            return v.evaluate() == self._dim
        raise TypeError(
            "Unable to compare a DimensionObject to {}".format(type(v)))

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

    def __gt__(self, obj):
        """
        usual
        """
        if isinstance(self._dim, int) and isinstance(obj._dim, int):
            return self._dim < obj._dim
        elif isinstance(self._dim, int) and isinstance(obj._dim, str):
            return False
        elif isinstance(self._dim, str) and isinstance(obj._dim, int):
            return True
        else:
            if self._dim == obj._dim:
                return False
            raise RuntimeError(
                "Cannot decide between {} and {}".format(self, obj))


class ShapeObject:
    """
    Handles mathematical operations around shapes.
    """

    def __init__(self, shape, dtype=None):
        """
        @param      shape       tuple or `numpy.array`
        @param      dtype       dtype
        """
        if isinstance(shape, numpy.ndarray):
            self._shape = [DimensionObject(s) for s in shape.shape]
            self._dtype = shape.dtype
        elif isinstance(shape, dict) and 'type' in shape:
            tshape = shape['type']
            if tshape['kind'] == 'tensor':
                self._shape = [DimensionObject(s) for s in tshape['shape']]
                self._dtype = tshape['elem']
            elif tshape['kind'] == 'map':
                self._shape = []
                self._dtype = 'map'
            else:
                raise ValueError("Wrong shape value {}".format(shape))
        elif isinstance(shape, (tuple, list)):
            self._shape = []
            for s in shape:
                self._shape.append(DimensionObject(s))
            self._dtype = dtype
        elif shape is None:
            # shape is unknown
            self._shape = None
            self._dtype = dtype
        else:
            raise TypeError(
                "Unexpected type for shape: {}".format(type(shape)))
        if self._dtype is None:
            raise ValueError(
                "dtype cannot be None, shape type is {}\n{}".format(
                    type(shape), shape))
        if self._shape is not None:
            for i, a in enumerate(self._shape):
                if not isinstance(a, DimensionObject):
                    raise TypeError('Dimension {} has a wrong type {}'.format(
                        i, type(a)))

    def copy(self, dtype=None):
        """
        A copy not a deepcopy.

        @param      dtype   None or a value to rewrite the type.
        @return             @see cl ShapeObject
        """
        return ShapeObject(self._shape.copy(),
                           self.dtype if dtype is None else dtype)

    def __getitem__(self, index):
        """
        Extracts a specific dimension.
        """
        return self._shape[index]

    def __setitem__(self, index, value):
        """
        Changes a specific dimension.
        """
        self._shape[index] = value

    @property
    def shape(self):
        """
        Returns the stored shape.
        """
        return tuple(self._shape)

    def __len__(self):
        """
        Returns the number of dimensions.
        """
        if self._shape is None:
            return 0
        return len(self._shape)

    @property
    def dtype(self):
        """
        Returns the stored *dtype*.
        """
        return self._dtype

    def reduce(self, axis=1, keepdims=False, dtype=None):
        """
        Reduces the matrix. Removes one dimension.

        @param      axis        axis
        @param      keepdims    keep dimensions, replaces the removed
                                dimension by 1
        @param      dtype       if not None, changes the type
        @return                 new dimension
        """
        if 0 <= axis < len(self._shape):
            cp = self._shape.copy()
            if keepdims:
                cp[axis] = DimensionObject(1)
            else:
                del cp[axis]
            return ShapeObject(cp, self._dtype if dtype is None else dtype)
        raise IndexError("axis={} is wrong, shape is {}".format(axis, self))

    def __repr__(self):
        """
        usual
        """
        st = str(self.dtype)
        if "'" in st:
            st = st.split("'")[1]
        st_shape = []
        for s in self.shape:
            if isinstance(s._dim, (int, str)):
                st_shape.append(str(s._dim))
            else:
                st_shape.append(repr(s))
        st_shape = '({})'.format(", ".join(st_shape))
        return "ShapeObject({}, dtype={})".format(st_shape, st)

    def __iter__(self):
        """
        Iterators over dimensions.
        """
        for d in self._shape:
            yield d

    def __gt__(self, a):
        """
        Compares shapes. Operator ``>``.
        """
        if len(self) > len(a):
            return True
        if len(self) < len(a):
            return False
        for d1, d2 in zip(self, a):
            if d1 > d2:
                return True
            if d1 < d2:
                return False
        return False

    def __eq__(self, a):
        """
        Tests equality between two shapes.
        """
        if len(self) != len(a):
            return False
        for d1, d2 in zip(self, a):
            if d1 == d2:
                continue
            return False
        return True

    def evaluate(self, **kwargs):
        """
        Evaluates the shape.
        """
        vs = []
        for v in self:
            d = v.evaluate(**kwargs)
            vs.append(d)
        return ShapeObject(tuple(vs), self._dtype)

    def product(self):
        """
        Multiplies all the dimension.

        @return     @see cl DimensionObject
        """
        cl = self[0]
        for i in range(1, len(self)):
            cl = cl * self[i]
        return cl

    def append(self, dim):
        """
        Appends a dimension.
        """
        if isinstance(dim, DimensionObject):
            self._shape.append(dim)
        else:
            self._shape.append(DimensionObject(dim))

    def squeeze(self, axis):
        """
        Removes one dimension.
        """
        cp = self.copy()
        cp.drop_axis(axis)
        return cp

    def transpose(self, perm):
        """
        Removes one dimension.
        """
        cp = self.copy()
        for i, p in enumerate(perm):
            cp._shape[i] = self._shape[p]
        return cp

    def drop_axis(self, axis):
        """
        Drops an axis.
        """
        if isinstance(axis, (tuple, list)):
            for i in sorted(axis, reverse=True):
                del self._shape[i]
        else:
            del self._shape[axis]
