"""
@file
@brief Shape object.
"""
import numpy


class BaseDimensionShape:
    """
    Base class to @see cl DimensionObject,
    @see cl ShapeOperator, @see cl ShapeObject.
    """

    def to_string(self, use_x=True):
        """
        Converts the object into a string.
        """
        raise NotImplementedError()

    def evaluate(self, **kwargs):
        """
        Evaluates the object, reduces the expression
        to a number or a string.
        """
        raise NotImplementedError()  # pragma: no cover


class ShapeOperator(BaseDimensionShape):
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
                    "All arguments must be of type DimensionObject not '{}'."
                    "".format(type(a)))

    def __repr__(self):
        """
        usual
        """
        return "{0}('{1}', {2}, '{2}', {3})".format(
            self.__class__.__name__, self._name,
            self._fct_string, self._args)

    def to_string(self, use_x=True):
        """
        Displays as a string.

        @return     a string
        """
        raise NotImplementedError(  # pragma: no cover
            "Operator '{}' does not implement 'to_string': {}.".format(
                self.__class__.__name__, repr(self)))

    def evaluate(self, **kwargs):
        """
        Evalutes the operator.

        @param  kwargs      value for the variables.
        @return             string or integer
        """
        args = []
        has_string = False
        for a in self._args:
            a = DimensionObject._same_(a)
            v = a.evaluate(**kwargs)
            if isinstance(v, str):
                has_string = True
            args.append(v)
        if has_string:
            res = self._evaluate_string_(args, **kwargs)
        else:
            try:
                res = self._fct(*args)
            except TypeError as e:
                raise RuntimeError(
                    "Unable to evaluate operator {} due to {}".format(repr(self), e)) from e
        return res

    def _evaluate_string_(self, args, **kwargs):
        """
        Evalutes the operator assuming some of them are still strings.

        @param  args        arguments extracted by method *evaluate*
        @param  kwargs      value for the variables.
        @return             string or integer
        """
        raise NotImplementedError(
            "This function must be overwritten.")  # pragma: no cover


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
            raise TypeError('x cannot be a tuple')  # pragma: no cover
        if isinstance(y, tuple):
            raise TypeError('y cannot be a tuple')  # pragma: no cover

    def _to_string1(self, x, y):
        return DimensionObject(self._fct(x._dim, y._dim)).to_string()

    def _to_string2(self, x, y):
        return DimensionObject("{}{}{}".format(x._dim, self._name, y._dim)).to_string()

    def _to_string2b(self, x, y):
        return DimensionObject("({}){}({})".format(x._dim, self._name, y._dim)).to_string()

    def _to_string3(self, x):
        return DimensionObject("{}{}x".format(x._dim, self._name)).to_string()

    def to_string(self, use_x=True):
        """
        Applies binary operator to a dimension.

        @param      use_x   use `'x'` if dimension is unknown
        @return             a string
        """
        x, y = self._args  # pylint: disable=W0632
        if isinstance(x._dim, int):
            if isinstance(y, DimensionObject):
                if isinstance(y._dim, int):
                    return self._to_string1(x, y)
                if isinstance(y._dim, str):
                    return self._to_string2(x, y)
                if y._dim is None:
                    if use_x:
                        return self._to_string3(x)
                    return DimensionObject("{}{}DimensionObject()".format(
                        x._dim, self._name)).to_string()
                raise TypeError(  # pragma: no cover
                    "Unable to handle type '{}'.".format(type(y._dim)))
            raise TypeError(  # pragma: no cover
                "Unable to handle type '{}'.".format(type(y)))
        elif isinstance(x._dim, str):
            if isinstance(y._dim, int):
                return self._to_string2(x, y)
            if isinstance(y._dim, str):
                return self._to_string2b(x, y)
            raise TypeError(  # pragma: no cover
                "Unable to handle type '{}'.".format(type(y._dim)))
        raise TypeError(  # pragma: no cover
            "Unable to handle type '{}'.".format(type(x._dim)))

    def _evaluate_string_(self, args, **kwargs):
        """
        Evalutes the operator assuming some of them are still strings.

        @param  args        arguments extracted by method *evaluate*
        @param  kwargs      value for the variables.
        @return             string or integer
        """
        return self._name.join(map(lambda s: '({})'.format(s), args))


class ShapeBinaryFctOperator(ShapeBinaryOperator):
    """
    Base class for shape binary operator defined by a function.
    """

    def _to_string2(self, x, y):
        return DimensionObject("{}({},{})".format(self._name, x._dim, y._dim)).to_string()

    def _to_string2b(self, x, y):
        return DimensionObject("{}({},{})".format(self._name, x._dim, y._dim)).to_string()

    def _to_string3(self, x):
        return DimensionObject("{}({},x)".format(self._name, x._dim)).to_string()

    def _evaluate_string_(self, args, **kwargs):
        """
        Evalutes the operator assuming some of them are still strings.

        @param  args        arguments extracted by method *evaluate*
        @param  kwargs      value for the variables.
        @return             string or integer
        """
        return "{}({})".format(self._name, ",".join(map(str, args)))


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


class ShapeOperatorGreater(ShapeBinaryOperator):
    """
    Shape comparison.
    """

    def __init__(self, x, y):
        ShapeBinaryOperator.__init__(
            self, '>', lambda a, b: a > b, 'lambda a, b: a > b', x, y)

    def __repr__(self):
        """
        Displays a string.

        @return             a string
        """
        return "{0}({1}, {2})".format(
            self.__class__.__name__, repr(self._args[0]), repr(self._args[1]))


class ShapeOperatorMax(ShapeBinaryFctOperator):
    """
    Best on each dimension.
    """

    def __init__(self, x, y):
        ShapeBinaryFctOperator.__init__(
            self, 'max', lambda a, b: max(a, b), 'max(a, b)', x, y)

    def __repr__(self):
        """
        Displays a string.

        @return             a string
        """
        return "{0}({1}, {2})".format(
            self.__class__.__name__, repr(self._args[0]), repr(self._args[1]))


class DimensionObject(BaseDimensionShape):
    """
    One dimension of a shape.
    """

    def __init__(self, obj):
        """
        @param  obj     int or @see cl DimensionObject or None to
                        specify something unknown
        """
        if obj is None or obj == 0 or obj == '?':
            self._dim = None
        elif isinstance(obj, (int, str, ShapeOperator, DimensionObject,
                              numpy.int32, numpy.int64)):
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

    def to_string(self, use_x=True):
        """
        Represents the dimension as a string.
        """
        if isinstance(self._dim, int):
            return '{}'.format(self._dim)
        if isinstance(self._dim, ShapeOperator):
            return self._dim.to_string()
        if isinstance(self._dim, str):
            return self._dim
        if self._dim is None:
            return 'x' if use_x else '?'
        raise NotImplementedError(  # pragma: no cover
            "Not implemented for '{}'.".format(repr(self)))

    def evaluate(self, **kwargs):
        """
        Evalutes the dimension.

        @param  kwargs      value for the variables.
        @return             string or integer
        """
        if isinstance(self._dim, (int, ShapeOperator, DimensionObject)):
            res = self._dim
        elif isinstance(self._dim, str):
            if self._dim in kwargs:
                res = kwargs[self._dim]
            else:
                res = self._dim
        elif self._dim is None:
            pref = str(hex(id(self)))[2:]
            res = "n{}".format(pref)
        elif isinstance(self._dim, ):
            res = self._dim.evaluate(**kwargs)
        else:
            raise NotImplementedError(  # pragma: no cover
                "Not implemented for '{}'.".format(repr(self)))
        if isinstance(res, (ShapeOperator, DimensionObject)):
            return res.evaluate(**kwargs)
        return res

    def __eq__(self, v):
        """
        usual
        """
        if isinstance(v, (int, str)):
            return self._dim == v
        if isinstance(v, DimensionObject):
            return v == self._dim
        if isinstance(v, ShapeOperator):
            ve = v.evaluate()
            return ve == self._dim
        if v is None:
            return self._dim is None
        raise TypeError(  # pragma: no cover
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
        if obj is None:
            return not isinstance(self._dim, int)
        if isinstance(self._dim, int) and isinstance(obj._dim, int):
            return self._dim > obj._dim
        return DimensionObject(
            ShapeOperatorGreater(self, DimensionObject._same_(obj)))


class ShapeObject(BaseDimensionShape):
    """
    Handles mathematical operations around shapes.
    It stores a type (:epkg:`numpy` type),
    and a name to somehow have an idea of where
    the shape comes from in the :epkg:`ONNX` graph.
    The shape itself is defined by a list of
    @see cl DimensionObject or @see cl ShapeOperator
    or *None* if the shape is unknown. A dimension is an
    integer or a variable encoded as a string. This variable
    is a way to tell the dimension may vary.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import numpy
        from mlprodict.onnxrt.shape_object import ShapeObject

        sh1 = ShapeObject((1, 2), dtype=numpy.float32)
        sh2 = ShapeObject((45, 2), dtype=numpy.float32)
        mx = max(sh1, sh2)
        print(mx)

        sh1 = ShapeObject((1, 2), dtype=numpy.float32)
        sh2 = ShapeObject((None, 2), dtype=numpy.float32)
        print(sh2)
        mx = max(sh1, sh2)
        print(mx.to_string())

        sh1 = ShapeObject((1, 2), dtype=numpy.float32)
        sh2 = ShapeObject(('n', 2), dtype=numpy.float32)
        print(sh2)
        mx = max(sh1, sh2)
        print(mx.evaluate(n=4))
    """

    def __init__(self, shape, dtype=None, use_n1=False, name=None):
        """
        @param      shape       tuple or `numpy.array`
        @param      dtype       dtype
        @param      use_n1      use `'n'` if the first dimension is unknown
        @param      name        optional, for debugging purposes
        """
        self.name = name
        if isinstance(shape, numpy.ndarray):
            self._shape = [DimensionObject(s) for s in shape.shape]
            self._dtype = shape.dtype
        elif isinstance(shape, dict) and 'type' in shape:
            tshape = shape['type']
            if tshape['kind'] == 'tensor':
                if tshape['shape'] == ('?', ):
                    self._shape = None
                else:
                    self._shape = [DimensionObject(s) for s in tshape['shape']]
                self._dtype = tshape['elem']
            elif tshape['kind'] == 'map':
                self._shape = []
                self._dtype = 'map'
            else:
                raise ValueError(  # pragma: no cover
                    "Wrong shape value {}".format(shape))
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
            raise TypeError(  # pragma: no cover
                "Unexpected type for shape: {}".format(type(shape)))

        if self._dtype is None:
            raise ValueError(
                "dtype cannot be None, shape type is {}\n{}".format(
                    type(shape), shape))
        if self._dtype in (float, 'double'):
            self._dtype = numpy.float64
        elif self._dtype in ('float32', 'float'):
            self._dtype = numpy.float32
        elif self._dtype in (numpy.float16, 'float16'):
            self._dtype = numpy.float16
        elif self._dtype in ('int32', ):
            self._dtype = numpy.int32
        elif self._dtype in (int, 'int', 'int64'):
            self._dtype = numpy.int64
        elif self._dtype in (str, 'str', numpy.str_):
            self._dtype = numpy.str_
        elif (hasattr(self._dtype, 'type') and self._dtype.type is numpy.string_):
            pass
        elif self._dtype in (bool, 'bool', numpy.bool_):
            self._dtype = numpy.bool_
        elif self._dtype in (object, numpy.object_):
            pass
        elif self._dtype in (numpy.int8, 'int8', ):
            self._dtype = numpy.int8
        elif self._dtype in (numpy.uint8, 'uint8', ):
            self._dtype = numpy.uint8
        elif self._dtype in (numpy.int16, 'int16', ):
            self._dtype = numpy.int16
        elif self._dtype in (numpy.uint16, 'uint16', ):
            self._dtype = numpy.uint16
        elif self._dtype in (numpy.uint32, 'uint32', ):
            self._dtype = numpy.uint32
        elif self._dtype in (numpy.uint64, 'uint64', ):
            self._dtype = numpy.uint64
        elif self._dtype not in {
                numpy.float32, numpy.float64, numpy.int32, numpy.int64,
                numpy.str_, numpy.bool_, numpy.float16, None,
                numpy.complex64, numpy.complex128,
                'map'}:
            raise ValueError(  # pragma: no cover
                "dtype has an unexpected value: '{}'.".format(self._dtype))
        if self._shape is not None:
            for i, a in enumerate(self._shape):
                if not isinstance(a, DimensionObject):
                    raise TypeError(  # pragma: no cover
                        'Dimension {} has a wrong type {}'.format(
                            i, type(a)))
            if use_n1:
                sh = self._shape[0] if self._shape else None
                if isinstance(sh, DimensionObject) and sh._dim is None:
                    sh._dim = 'n'
        if self._shape is not None:
            for s in self._shape:
                if isinstance(s, int):
                    raise TypeError(  # pragma: no cover
                        "Unexpected type int in shape %r." % self)

    def reshape(self, shape):
        """
        Creates a new shape, checks the number of elements is the same.
        """
        sh = ShapeObject(shape, self.dtype, getattr(self, '_dim', None),
                         self.name)
        p1 = self.product().evaluate()
        p2 = sh.product().evaluate()
        if isinstance(p1, int) and p1 != p2:
            raise ValueError("Shape {} cannot be reshaped into {} "
                             "(p1={}, p2={}).".format(sh, shape, p1, p2))
        return sh

    def copy(self, dtype=None, name=None):
        """
        A copy not a deepcopy.

        @param      dtype   None or a value to rewrite the type.
        @param      name    overwrites the name
        @return             @see cl ShapeObject
        """
        if self._shape is None:
            return ShapeObject(None, dtype=self.dtype, name=name or self.name)
        return ShapeObject(self._shape.copy(),
                           self.dtype if dtype is None else dtype,
                           name=name or self.name)

    def __getitem__(self, index):
        """
        Extracts a specific dimension.
        """
        if self._shape is None:
            return None
        if isinstance(index, int) and index >= len(self._shape):
            return 1
        return self._shape[index]

    def __setitem__(self, index, value):
        """
        Changes a specific dimension.
        """
        if self._shape is None:
            return
        while len(self._shape) <= index:
            self._shape.append(DimensionObject(1))
        self._shape[index] = value

    @property
    def shape(self):
        """
        Returns the stored shape.
        """
        if self._shape is None:
            return None
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
        if self._shape is None:
            if self.name is None:
                return self.copy()
            return self.copy(name="{}-RD".format(self.name))
        if axis is None:
            return ShapeObject((1, ), self._dtype if dtype is None else dtype,
                               name="{}-RDN".format(self.name))

        if isinstance(axis, ShapeObject):

            def drop_axis(shape, a):
                c = list(shape)
                del c[a[0]]
                return c

            return ShapeObjectFct(
                drop_axis, self, axis, name="DropAxis", dtype=self.dtype)

        if 0 <= axis < len(self._shape):
            cp = self._shape.copy()
            if keepdims:
                cp[axis] = DimensionObject(1)
            else:
                del cp[axis]
            return ShapeObject(cp, self._dtype if dtype is None else dtype,
                               name="{}-RD".format(self.name))
        raise IndexError("axis={} is wrong, shape is {}-tuple and equal to "
                         "{}".format(axis, len(self._shape), self))

    def __repr__(self):
        """
        usual
        """
        st = str(self.dtype)
        if "'" in st:
            st = st.split("'")[1]

        if self.shape is None:
            if self.name is None:
                return "ShapeObject(None, dtype={})".format(st)
            return "ShapeObject(None, dtype={}, name='{}')".format(st, self.name)

        st_shape = []
        for s in self.shape:
            if isinstance(getattr(s, "_dim", None), (int, str)):
                st_shape.append(str(s._dim))
            else:
                st_shape.append(repr(s))
        if len(st_shape) == 1:
            st_shape.append('')
        st_shape = '({})'.format(", ".join(st_shape))
        if self.name is None:
            return "ShapeObject({}, dtype={})".format(st_shape, st)
        return "ShapeObject({}, dtype={}, name='{}')".format(
            st_shape, st, self.name)

    def __iter__(self):
        """
        Iterators over dimensions.
        """
        if self._shape is not None:
            for d in self._shape:
                yield d

    def __gt__(self, a):
        """
        Compares shapes. Operator ``>``.
        """
        if isinstance(a, tuple):
            a = ShapeObject(a, dtype=self._dtype)
        if self._shape is None and a._shape is None:
            return False
        if self._shape is None:
            return True
        if a._shape is None:
            return False
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
        if isinstance(a, tuple):
            a = ShapeObject(a, dtype=self._dtype)
        if self._shape is None and a._shape is None:
            return True
        if self._shape is None or a._shape is None:
            return False
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
        return ShapeObject(tuple(vs), self._dtype, name="{}-EV".format(self.name))

    def to_string(self, use_x=False):
        """
        Converts shapes into a string.
        """
        shapes = []
        for a in self._shape:
            shapes.append(a.to_string(use_x=use_x))
        return '({})'.format(', '.join(shapes))

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
        if self._shape is None:
            return
        if isinstance(dim, DimensionObject):
            self._shape.append(dim)
        else:
            self._shape.append(DimensionObject(dim))

    def insert(self, dim, pos=0):
        """
        Inserts a dimension at position *pos*.
        """
        if self._shape is None:
            return
        if isinstance(dim, DimensionObject):
            self._shape.insert(pos, dim)
        else:
            self._shape.insert(pos, DimensionObject(dim))

    def squeeze(self, axis):
        """
        Removes one dimension.
        """
        cp = self.copy(name='{}-SZ'.format(self.name))
        cp.drop_axis(axis)
        return cp

    def unsqueeze(self, axes):
        """
        Adds dimensions.
        """
        cp = self
        name = '{}-USZ'.format(self.name)
        for ax in axes[::-1]:
            cp = cp.copy(name=name)
            cp.insert(ax, 1)
        return cp

    def transpose(self, perm):
        """
        Removes one dimension.
        """
        if self.shape is None:
            return self.copy(name='{}-TR'.format(self.name))
        cp = ShapeObject([None for p in perm], dtype=self.dtype,
                         name="{}-TR".format(self.name))
        for i, p in enumerate(perm):
            if p >= len(self):
                # This should not happen.
                cp._shape[i] = None
            else:
                cp._shape[i] = self._shape[p]
        return cp

    def drop_axis(self, axis):
        """
        Drops an axis.
        """
        if self._shape is not None:
            if isinstance(axis, (tuple, list)):
                for i in sorted(axis, reverse=True):
                    del self._shape[i]
            else:
                del self._shape[axis]

    def broadcast(self, a):
        """
        Computes the shape after a broadcast.
        """
        if a is None:
            raise ValueError("a should not be None")  # pragma: no cover
        if a._shape is None:
            return a.copy()
        if self._shape is None:
            return self.copy()
        mx = max(len(self._shape), len(a._shape))
        res = []
        for i in range(mx):
            if i < len(self._shape):
                if i < len(a._shape):
                    res.append(ShapeOperatorMax(self[i], a[i]))
                else:
                    res.append(self[i])
            else:
                res.append(a[i])
        return ShapeObject(tuple(res), self.dtype, False,
                           name="broadcast-{}-{}".format(self.name, a.name))

    @staticmethod
    def _infer_merged_type(*args, use_dtype=True):
        if use_dtype:
            tys = set(a.dtype for a in args)
        else:
            tys = set(args)
        if len(tys) == 1:
            return list(tys)[0]
        if any(tys & {numpy.float64, numpy.int64,
                      numpy.float32, numpy.int32,
                      numpy.float16}):
            return numpy.float64
        raise RuntimeError(  # pragma: no cover
            "Unable to infer types based on {} ({}).".format(
                tys, len(tys)))

    def concat_columns(self, axis, *shapes):
        """
        Concatenates columns from *shapes* to this one
        along one axis.
        """
        args = [self] + list(shapes)
        dtype = self._infer_merged_type(*args)
        dim_axis = self[axis]
        if isinstance(dim_axis, int):
            dim_axis = DimensionObject(dim_axis)
        if dim_axis is None:
            return ShapeObject(None, dtype=dtype)
        if isinstance(dim_axis, int):
            raise TypeError(  # pragma: no cover
                "Unexpected type for shape %r." % self)
        for a in shapes:
            if a[axis] is None:
                return ShapeObject(None, dtype=dtype)
            dim_axis = dim_axis + a[axis]
        a0 = args[0].copy(dtype=dtype)
        a0[axis] = dim_axis
        return a0

    @staticmethod
    def einsum_shape(equation, *inputs):
        """
        Computes :epkg:`einsum` shapes.
        Not the most efficient one as it creates variables
        of the given shapes.
        """
        for inp in inputs:
            if inp.shape is None:
                return inp
        inp, out = [_.strip() for _ in equation.split(b"->")]
        inps = [_.strip() for _ in inp.split(b',')]
        if len(inputs) != len(inps):
            raise RuntimeError(  # pragma: no cover
                "Input mismatch between '{}' and {}.".format(equation, inps))
        shs = {}
        for a, b in zip(inps, inputs):
            if len(a) != len(b):
                raise RuntimeError(  # pragma: no cover
                    "Input mismatch '{}' (in '{}') and {}.".format(a, equation, b))
            for c, s in zip(a, b):
                if c not in shs:
                    shs[c] = s
                elif shs[c] != s:
                    raise RuntimeError(  # pragma: no cover
                        "Equation '{}'. Dimension mismatch '{}' != {}.".format(
                            equation, s, shs[c]))
        new_shape = [shs[i] for i in out]
        return ShapeObject(new_shape, dtype=ShapeObject._infer_merged_type(*inputs))

    @staticmethod
    def gather_shape(input, indices, axis):
        """
        Computes Gather shapes.
        """
        input_rank = len(input)
        if input_rank is None:
            return ShapeObject(None, dtype=input._dtype)
        index_rank = len(indices)
        if index_rank is None:
            return ShapeObject(None, dtype=input._dtype)

        if axis < 0:
            axis = input_rank + axis

        shape = []
        for i in range(axis):
            shape.append(input[i])

        for dim in indices:
            shape.append(dim)

        for i in range(axis + 1, input_rank):
            shape.append(input[i])

        return ShapeObject(shape, dtype=input._dtype)


class ShapeObjectFct(ShapeObject):
    """
    Computes a shape depending on a user defined function.
    See @see cl Conv for an example.
    """

    def __init__(self, fct, *shapes, dtype=None, name=None):
        """
        @param      fct         function
        @param      shapes      shapes sent to fct
        @param      dtype       dtype
        @param      name        optional, for debugging purposes
        """
        ShapeObject.__init__(self, None, dtype=dtype, name=name)
        self._fct = fct
        self._shapes = shapes

    def evaluate(self, **kwargs):
        """
        Evaluates the shape.
        """
        vs = []
        for v in self._shapes:
            d = v.evaluate(**kwargs)
            vs.append(d)
        res = self._fct(*vs)
        if self.name is not None:
            res.name = self.name
        return res
