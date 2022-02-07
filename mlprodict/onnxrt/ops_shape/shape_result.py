"""
@file
@brief Class ShapeResult
"""
from enum import Enum


class ShapeInferenceException(RuntimeError):
    """
    Raised when shape inference fails.
    """
    pass


class OnnxKind(Enum):
    """
    Describes a result type.
    """
    Tensor = 0
    Sequence = 0
    Map = 0


class ShapeConstraint:
    """
    One constraint.

    :param name: variable name
    :param values: set of possible values
    """

    def __init__(self, name, values):
        if name == '?':
            raise ValueError("Name cannot be '?'.")
        self.name = name
        self.values = values

    def __eq__(self, other):
        "usual"
        if self.name != other.name:
            return False
        if self.values != other.values:
            return False
        return True

    def __repr__(self):
        "usual"
        return "%s(%r, %r)" % (
            self.__class__.__name__, self.name, self.values)

    def merge(self, cst):
        """
        Merges this constraint with *cst* into this one.
        """
        if isinstance(cst, list):
            for c in cst:
                self.merge(c)
            return
        self.values = self.values.intersection(cst.values)


class ShapeConstraintList:
    """
    A list of ShapeConstraint.
    """

    def __init__(self):
        self.csts = []

    def __contains__(self, cst):
        for a in self.csts:
            if cst == a:
                return True
        return False

    def append(self, cst):
        "Appends a new constraint to the list."
        self.csts.append(cst)

    def __repr__(self):
        return "ShapeConstraintList(%r)" % self.csts

    def __iter__(self):
        for c in self.csts:
            yield c

    def __len__(self):
        return len(self.csts)


class ShapeResult:
    """
    Contains information about shape and type of a result
    in an onnx graph.

    :param shape: shape if the result is a tensor
    :param dtype: element type if the result is a tensor
    :param sparse: is a the tensor sparse
    :param mtype: kind of the result (see class @see cl OnnxKind)
    :param constraints: list of constraints applying on variables
    """

    def __init__(self, shape=None, dtype=None, sparse=False,
                 mtype=OnnxKind.Tensor, constraints=None):
        self.mtype = mtype
        self.shape = list(shape)
        self.dtype = dtype
        self.sparse = sparse
        for i in range(0, len(self.shape)):  # pylint: disable=C0200
            if shape[i] in ('', None, '?'):
                raise ValueError(
                    "All dimensions must an int or a variable name, "
                    "%s is not." % (shape, ))
        if constraints is None:
            self.constraints = ShapeConstraintList()
        elif isinstance(constraints, ShapeConstraintList):
            self.constraints = constraints
        else:
            raise TypeError(
                "constraints must be of type(ShapeConstraintList).")

    def __repr__(self):
        """
        Usual
        """
        return "%s(%r, %r, %r, %r, constraints=%r)" % (
            self.__class__.__name__, self.shape, self.dtype,
            self.sparse, self.mtype, self.constraints)

    def __eq__(self, shape):
        """
        Tells if two shapes are identical.
        """
        return (self.mtype == shape.mtype and self.shape == shape.shape and
                self.dtype == shape.dtype and self.sparse == shape.sparse)

    def n_dims(self):
        """
        Returns the number of dimensions if it is a tensor.
        Raises an exception otherwise.
        """
        if self.mtype != OnnxKind.Tensor:
            raise ShapeInferenceException(
                "This shape is not a tensor %r." % self)
        return len(self.shape)

    def merge(self, other_result):
        """
        Merges constraints from *other_results* into *self*.
        """
        if (self.mtype != other_result.mtype or
                self.dtype != other_result.dtype or
                self.sparse != other_result.sparse):
            raise RuntimeError(
                "Unable to merge %r and %r." % (self, other_result))
        if len(self.shape) != len(other_result.shape):
            raise RuntimeError(
                "Length mismatch, unable to merge %r and %r." % (
                    self, other_result))
        updated = False
        if other_result.constraints is not None:
            for c in other_result.constraints:
                if c not in self.constraints:
                    self.constraints.append(c)
                    updated = True
        for a, b in zip(self.shape, other_result.shape):
            if a == b:
                continue
            if isinstance(a, int) and isinstance(b, int):
                raise RuntimeError(
                    "Inconsistancy between %r and %r." % (
                        self, other_result))
            elif isinstance(a, str):
                c = ShapeConstraint(a, {b})
                if c not in self.constraints:
                    updated = True
                    self.constraints.append(c)
            elif isinstance(b, str):
                c = ShapeConstraint(b, {a})
                if c not in self.constraints:
                    updated = True
                    self.constraints.append(c)
            else:
                raise NotImplementedError(
                    "Merge not implemented between %r and %r." % (
                        self, other_result))
        if len(self.constraints) > 4:
            raise RuntimeError(
                "The number of constraints should not that many (%r)." % (
                    self.constraints))
        return updated

    @staticmethod
    def broadcast(sh1, sh2):
        """
        Broadcasts dimensions for an element wise operator.

        :param sh1: ShapeResult
        :param sh2: ShapeResult
        :return: ShapeResult
        """
        if not isinstance(sh1, ShapeResult):
            raise TypeError("Unexpected type for sh1 %r." % type(sh1))
        if not isinstance(sh2, ShapeResult):
            raise TypeError("Unexpected type for sh2 %r." % type(sh2))
        if sh1.mtype != OnnxKind.Tensor:
            raise TypeError("sh1 must be a tensor not %r." % sh1.mtype)
        if sh2.mtype != OnnxKind.Tensor:
            raise TypeError("sh2 must be a tensor not %r." % sh2.mtype)
        if sh1.n_dims() != sh2.n_dims():
            raise ShapeInferenceException(
                "Broadcasting is only implemented for shape of the same "
                "size, shapes are %r and %r." % (sh1, sh2))
        if sh1.dtype != sh2.dtype:
            raise ShapeInferenceException(
                "Cannot broadcast shapes %r and %r (dtypes)."
                "" % (sh1, sh2))

        constraints = ShapeConstraintList()
        shape = []
        for a, b in zip(sh1.shape, sh2.shape):
            if isinstance(a, int) and isinstance(b, int):
                if a != b:
                    if min(a, b) == 1:
                        d = max(a, b)
                    else:
                        raise ShapeInferenceException(
                            "Cannot broadcast shapes %r and %r (dimensions)."
                            "" % (sh1, sh2))
                else:
                    d = a
            elif isinstance(a, int):
                if a != 1:
                    d = a
                    constraints.append(ShapeConstraint(b, {1, a}))
                else:
                    d = b
            elif isinstance(b, int):
                if b != 1:
                    d = b
                    constraints.append(ShapeConstraint(a, {1, b}))
                else:
                    d = a
            elif a == b:
                d = a
            else:
                raise ShapeInferenceException(
                    "Cannot broadcast shapes %r and %r." % (sh1, sh2))
            shape.append(d)
        res = ShapeResult(shape, sh1.dtype, sh1.sparse or sh2.sparse,
                          sh1.mtype, constraints)
        return res
