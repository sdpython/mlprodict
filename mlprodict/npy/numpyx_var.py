"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy
from onnx import (  # pylint: disable=E0611
    FunctionProto, ModelProto, TensorProto)
from onnx.helper import np_dtype_to_tensor_dtype
from .numpyx_types import (
    ParType, SequenceType, TensorType)


DEFAULT_OPSETS = {'': 18, 'ai.onnx.ml': 3}


class Par:
    """
    Defines a named parameter.

    :param name: parameter name
    :param dtype: parameter type (int, str, float)
    :param value: value of the parameter if known
    :param parent_op: node type it belongs to
    """

    def __init__(self, name: str, dtype: ParType, value: Optional[Any] = None,
                 parent_op: Optional[Tuple[str, str, int]] = None):
        if not isinstance(dtype, ParType):
            raise TypeError(
                f"dtype for parameter {name!r} must be of "
                f"ParType not {type(dtype)}.")
        if parent_op is None:
            raise ValueError(
                f"parent_op must be filled for paramenter {name!r}.")
        self.name = name
        self.dtype = dtype
        self.value = value
        self.parent_op = parent_op

    def __repr__(self):
        "usual"
        if self.value is None:
            return (
                f"{self.__class__.__name__}({self.name!r}, {self.dtype!r}, "
                f"parent_op={self.parent_op!r})")
        return (
            f"{self.__class__.__name__}"
            f"({self.name!r}, {self.dtype!r}, {self.value!r}, "
            f"parent_op={self.parent_op!r})")

    @property
    def onnx_type(self):
        "Returns the corresponding onnx type."
        return self.dtype.onnx_type

    def __eq__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __neq__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __lt__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __gt__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __le__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __ge__(self, x):
        "Should not be used."
        raise NotImplementedError()


class ManyIdentity:
    """
    Holds several instances of :class:`Var`.
    """

    def __init__(self, *inputs, input_indices=None):
        self.inputs = inputs
        self.onnx_op = None
        if input_indices is None:
            self.input_indices = [0 for i in self.inputs]
        else:
            self.input_indices = input_indices
        self.n_var_outputs = len(self.inputs)
        self.onnx_op_kwargs = {}
        self._prefix = "ManyIdentity_"

    def __repr__(self) -> str:
        "usual"
        args = list(map(repr, self.inputs))
        if max(self.input_indices) > 0:
            args.append(f"input_indices={self.input_indices}")
        s = ", ".join(args)
        return f"{self.__class__.__name__}({s})"

    def __len__(self):
        "Returns the number of merged variables."
        return len(self.inputs)

    def __getitem__(self, i):
        "Returns the ith elements."
        return self.inputs[i]

    def to_onnx(self, target_opsets: Optional[Dict[str, int]] = None,
                as_function: bool = False,
                name: Optional[str] = None,
                domain: Optional[str] = None,
                attributes: Optional[List[str]] = None,
                constraints: Optional[Dict[Any, TensorType]] = None
                ) -> Union[ModelProto, FunctionProto, List[Any]]:
        """
        Converts the recursive graph to ONNX.

        :param target_opsets: dictionary `{opset: version}`
        :param as_function: conversion to :class:`onnx.FunctionProto`
            or :class:`onnx.ModelProto`
        :param name: function name if *as_function* is True
        :param domain: function domain if *as_function* is True
        :param attributes: function attributes if any
        :param constraints: specifies a precise type for the type
            constraints when a function allows more than one type,
            this works if there is only one variable to be converted
        :return: ModelProto, FunctionProto
        """
        from .numpyx_graph_builder import _GraphBuilder

        # Var.to_onnx
        g = _GraphBuilder(target_opsets, as_function=as_function,
                          name=name, domain=domain, attributes=attributes,
                          constraints=constraints)
        done = set()
        outputs = []
        for var in self.inputs:
            vs = var._get_vars()
            for var in vs:
                key = id(var)
                if key in done:
                    continue
                g.append(var)
                done.add(key)
            outputs.append(vs[-1])
        onx = g.to_onnx(output_vars=outputs)
        if as_function:
            if len(outputs) != len(onx.output):
                raise RuntimeError(
                    f"Mismatch number of outputs, expecting {len(outputs)}, "
                    f"got ({len(onx.output)}).")
            if len(g.functions_) > 0:
                return [g.functions_, onx]
            return onx

        if len(outputs) != len(onx.graph.output):
            raise RuntimeError(
                f"Mismatch number of outputs, expecting {len(outputs)}, "
                f"got ({len(onx.graph.output)}).")
        return onx


class Var:
    """
    Defines a variable, a result...

    :param inputs: list of inputs
    :param op: apply on operator on the inputs
    :param inline: True to reduce the use of function and inline
        small functions, this only applies if *op* is a function
    :param n_var_outputs: number of the operator outputs
    :param input_indices: to select a specific output from the input
        operator
    :param kwargs: operator attributes

    Private attribute:

    :param onnx_input_type_: names given to the variables
    """

    def __init__(self, *inputs: List[Any],
                 op: Union[Callable, str, Tuple[str, str]] = None,
                 dtype: TensorType = None,
                 inline: bool = False,
                 n_var_outputs: Optional[int] = 1,
                 input_indices: Optional[List[int]] = None,
                 **kwargs):
        self.inputs = inputs
        self.n_var_outputs = n_var_outputs
        self.inline = inline
        if op is None:
            self.onnx_op = None  # a constant
        elif isinstance(op, tuple):
            self.onnx_op = op  # domain, operator name
        elif isinstance(op, str):
            self.onnx_op = ('', op)  # operator name
        else:
            self.onnx_op = (None, op)  # function to call
        self.onnx_op_kwargs = kwargs
        self._prefix = None
        self.dtype = dtype
        for i, inp in enumerate(self.inputs):
            if isinstance(inp, type):
                raise TypeError(f"Unexpected type for input {i} - {inp}.")
            if not isinstance(inp, numpy.ndarray):
                continue
            if (inp.size > 0 and
                    isinstance(inp.ravel()[0], (numpy.ndarray, Var))):
                raise TypeError(  # pragma: no cover
                    f"Unexpected type for input {i}: {type(inp)}, "
                    f"{inp.ravel()[0]}, op={op!r}")
        if input_indices is None:
            self.input_indices = [0 for i in self.inputs]
        elif not isinstance(input_indices, list):
            raise TypeError(
                f"input_indices is {type(input_indices)} "
                f"but len(inputs)={len(inputs)}.")
        else:
            self.input_indices = input_indices
        if len(self.input_indices) != len(self.inputs):
            raise RuntimeError(
                f"length mismatch len(self.input_indices)="
                f"{len(self.input_indices)} != len(self.inputs)="
                f"{len(self.inputs)}.")
        if self.onnx_op is None:
            if not isinstance(self, (Input, Cst)):
                raise RuntimeError(f"This case is not allowed: {self!r}.")

    def replace_inputs(self, new_inputs: List["Var"],
                       input_indices: Optional[List[int]] = None) -> "Var":
        """
        Replaces inputs by new ones. It creates a copy.
        It is needed when inlining functions.
        """
        new_var = Var(*new_inputs,
                      op=self.onnx_op,
                      dtype=self.dtype,
                      inline=self.inline,
                      input_indices=input_indices,
                      n_var_outputs=self.n_var_outputs,
                      **self.onnx_op_kwargs)
        new_var._prefix = self._prefix
        return new_var

    def __repr__(self):
        "usual"
        args = []
        for inp in self.inputs:
            n = inp.__class__.__name__
            args.append(f"{n[0]}.")
        if self.onnx_op is not None:
            args.append(f"op={self.onnx_op!r}")
        if self.n_var_outputs != 1:
            args.append(f"n_var_outputs={self.n_var_outputs!r}")
        if max(self.input_indices) != 0:
            args.append(f"input_indices={self.input_indices!r}")
        for k, v in sorted(self.onnx_op_kwargs.items()):
            args.append(f"{k}={v!r}")
        res = f"{self.__class__.__name__}({', '.join(args)})"
        return res

    def set_onnx_name(self, prefix: str):
        """
        Forces this variable to get this name during

        :param prefix: prefix
        """
        self._prefix = prefix

    def _get_vars(self):
        vs = []
        stack = [self]
        replacement = {}
        deleted = []
        while len(stack) > 0:
            var = stack.pop()
            key = id(var)
            if key in replacement:
                while key in replacement:
                    var = replacement[key]
                    key = id(var)
            if (var.onnx_op is not None and
                    var.onnx_op[0] is None and
                    var.inline):
                fct = var.onnx_op[1]
                applied = fct(*var.inputs, **var.onnx_op_kwargs)
                if isinstance(applied, (ManyIdentity, Var)):
                    stack.append(applied)
                    replacement[id(var)] = applied
                    deleted.append(var)
                    continue
                raise TypeError(
                    f"Unexpected type {type(applied)} as output of "
                    f"function {fct}.")
            vs.append(var)
            for i in reversed(var.inputs):
                if isinstance(i, Var):
                    stack.insert(0, i)
        res = list(reversed(vs))

        # replacement
        new_res = []
        for r in res:
            new_inputs = []
            new_indices = []
            repl = False
            for v, ind in zip(r.inputs, r.input_indices):
                key = id(v)
                if key in replacement:
                    while key in replacement:
                        var = replacement[key]
                        key = id(var)
                    new_inputs.append(var)
                    new_indices.append(ind)
                    repl = True
                else:
                    new_inputs.append(v)
                    new_indices.append(ind)
            if repl:
                new_r = r.replace_inputs(new_inputs, input_indices=new_indices)
                replacement[id(r)] = new_r
                new_res.append(new_r)
            else:
                new_res.append(r)

        # check the graph is consistent
        known = {}
        for r in new_res:
            known[id(r)] = r
            if isinstance(r, (Cst, Input)):
                continue
            for ind, i in enumerate(r.inputs):
                if id(i) not in known:
                    raise RuntimeError(
                        f"An input {ind} ({id(i)}) from {id(r)}-{r} "
                        f"is not known, it is not produced by a "
                        f"previous var (scheduled for replacement: "
                        f"{id(i) in replacement}).")
        return new_res

    @property
    def is_function(self):
        """
        Tells if this variable encapsulate a function.
        """
        return self.onnx_op is not None and self.onnx_op[0] is None

    def to_onnx(self, target_opsets: Optional[Dict[str, int]] = None,
                as_function: bool = False,
                name: Optional[str] = None,
                domain: Optional[str] = None,
                attributes: Optional[List[str]] = None,
                constraints: Optional[Dict[Any, TensorType]] = None
                ) -> Union[ModelProto, FunctionProto, List[Any]]:
        """
        Converts the recursive graph to ONNX.

        :param target_opsets: dictionary `{opset: version}`
        :param as_function: conversion to :class:`onnx.FunctionProto`
            or :class:`onnx.ModelProto`
        :param name: function name if *as_function* is True
        :param domain: function domain if *as_function* is True
        :param attributes: function attributes if any
        :param constraints: specifies a precise type for the type
            constraints when a function allows more than one type,
            this works if there is only one variable to be converted
        :return: ModelProto, FunctionProto
        """
        from .numpyx_graph_builder import _GraphBuilder

        # Var.to_onnx
        if target_opsets is None:
            target_opsets = DEFAULT_OPSETS
        g = _GraphBuilder(target_opsets, as_function=as_function,
                          name=name, domain=domain, attributes=attributes,
                          constraints=constraints)
        vs = self._get_vars()
        for var in vs:
            g.append(var)
        onx = g.to_onnx()
        if as_function and len(g.functions_) > 0:
            return [g.functions_, onx]
        return onx

    # Operators

    def _binary_op(self, ov, op_name, **kwargs):
        from .numpyx_core_api import var
        if isinstance(ov, (int, float, numpy.ndarray, Cst)):
            return var(self, var(ov, self, op='CastLike'), op=op_name)
        return var(self, ov, op=op_name, **kwargs)

    def __neg__(self):
        """
        Automatically adds operator `Neg` to the graph.
        It does not cast automatically.
        """
        from .numpyx_core_api import var
        return var(self, op="Neg")

    def __invert__(self):
        """
        Automatically adds operator `BitwiseNot` to the graph.
        It does not cast automatically.
        """
        from .numpyx_core_api import var
        return var(self, op="BitwiseNot")

    def __add__(self, ov):
        """
        Automatically adds operator `Add` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'Add')

    def __sub__(self, ov):
        """
        Automatically adds operator `Sub` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'Sub')

    def __mul__(self, ov):
        """
        Automatically adds operator `Mul` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'Mul')

    def __matmul__(self, ov):
        """
        Automatically adds operator `MatMul` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'MatMul')

    def __truediv__(self, ov):
        """
        Automatically adds operator `Div` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'Div')

    def __mod__(self, ov):
        """
        Automatically adds operator `Mod` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'Mod')

    def __pow__(self, ov):
        """
        Automatically adds operator `Pow` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'Pow')

    def __lt__(self, ov):
        """
        Automatically adds operator `Less` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'Less')

    def __le__(self, ov):
        """
        Automatically adds operator `LessOrEqual` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'LessOrEqual')

    def __gt__(self, ov):
        """
        Automatically adds operator `Greater` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'Greater')

    def __ge__(self, ov):
        """
        Automatically adds operator `GreaterOrEqual` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'GreaterOrEqual')

    def __eq__(self, ov):
        """
        Automatically adds operator `Equal` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'Equal')

    def __ne__(self, ov):
        """
        Automatically adds operator `Not + Equal` to the graph.
        It does not cast automatically.
        """
        from .numpyx_core_api import var
        return var(self._binary_op(ov, 'Equal'), op="Not")

    def __lshift__(self, ov):
        """
        Automatically adds operator `BitShift` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'BitShift', direction="LEFT")

    def __rshift__(self, ov):
        """
        Automatically adds operator `BitShift` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'BitShift', direction="RIGHT")

    def __and__(self, ov):
        """
        Automatically adds operator `BitwiseAnd` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'BitwiseAnd')

    def __or__(self, ov):
        """
        Automatically adds operator `BitwiseOr` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'BitwiseOr')

    def __xor__(self, ov):
        """
        Automatically adds operator `BitwiseXor` to the graph.
        It does not cast automatically.
        """
        return self._binary_op(ov, 'BitwiseXor')

    @property
    def T(self):
        "Transpose."
        from .numpyx_core_api import var
        return var(self, op='Transpose', perm=[1, 0])

    def astype(self, dtype):
        "Cast"
        from .numpyx_core_api import var
        if not isinstance(dtype, int):
            try:
                dtype = np_dtype_to_tensor_dtype(dtype)
            except KeyError:  # pylint: disable=E1101
                if dtype == numpy.float32:
                    dtype = TensorProto.FLOAT
                elif dtype == numpy.float64:
                    dtype = TensorProto.DOUBLE
                elif dtype == numpy.int64:
                    dtype = TensorProto.INT64
                elif dtype == numpy.int32:
                    dtype = TensorProto.INT32
                elif dtype == numpy.int16:
                    dtype = TensorProto.INT16
                elif dtype == numpy.int8:
                    dtype = TensorProto.INT8
                elif dtype == numpy.uint64:
                    dtype = TensorProto.UINT64
                elif dtype == numpy.uint32:
                    dtype = TensorProto.UINT32
                elif dtype == numpy.uint16:
                    dtype = TensorProto.UINT16
                elif dtype == numpy.uint8:
                    dtype = TensorProto.UINT8
                elif dtype == numpy.float16:
                    dtype = TensorProto.FLOAT16
                elif dtype in (bool, numpy.bool_):
                    dtype = TensorProto.BOOL
                elif dtype in (str, numpy.str_):
                    dtype = TensorProto.STRING
                else:
                    raise RuntimeError(  # pragma: no cover
                        f"Unable to guess type for dtype={dtype}.")

        return var(self, op="Cast", to=dtype)

    @property
    def shape(self):
        "Shape"
        from .numpyx_core_api import var
        return var(self, op='Shape')

    def reshape(self, shape):
        "Reshape"
        from .numpyx_core_api import var
        if isinstance(shape, (tuple, list)):
            shape = numpy.array(shape, dtype=numpy.int64)
        return var(self, shape, op="Reshape")

    def sum(self, axis=None, keepdims=0):
        "See :func:`numpy.sum`."
        from .numpyx_core_api import var
        if axis is None:
            return var(self, op="ReduceSum", keepdims=keepdims)
        if isinstance(axis, int):
            axis = [axis]
        if isinstance(axis, (tuple, list)):            
            from .numpyx_core_api import cst
            axis = cst(numpy.array(axis, dtype=numpy.int64))
        return var(self, axis, op="ReduceSum", keepdims=keepdims)

    def copy(self):
        """
        Returns a copy of self (use of Identity node).
        """
        from .numpyx_core_api import var
        return var(self, op="Identity")

    def flatten(self):
        """
        Flattens a matrix (see :epkg:`numpy:ndarray:flatten`).

        :param axis: only flatten from axis to the end.
        :return: :class:`Var`
        """
        from .numpyx_core_api import var
        return var(self, op="Flatten")

    def get(self, index: int) -> "Var":
        """
        If an operator or a function returns more than one output,
        this takes only one.

        :param index: index of the output to select
        :return: Var
        """
        if index < 0 or index >= self.n_var_outputs:
            raise ValueError(
                f"index={index} must be positive and < {self.n_var_outputs} "
                f"for var={self!r}.")
        return Var(self, input_indices=[index], op="Identity")

    def __getitem__(self, index: Any) -> "Var":
        """
        Implements indexing.
        """
        if self.n_var_outputs != 1:
            # Multioutut
            if not isinstance(index, int):
                raise TypeError(
                    f"Only indices are allowed when selecting an output, "
                    f"not {type(index)}).")
            return self.get(index)
        raise NotImplementedError("indexing is not implemented yet.")


class Input(Var):
    """
    Defines an input, a placeholder.

    :param name: input name or None if undefined
    """

    def __init__(self, name=None):
        Var.__init__(self)
        self.name = name
        self._prefix = name or "I"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"


class Cst(Var):
    """
    Defines a constant.
    """

    def __init__(self, cst: Any):
        if isinstance(cst, numpy.ndarray):
            Var.__init__(self, cst, op="Identity")
        elif isinstance(cst, int):
            Var.__init__(self, numpy.array([cst], dtype=numpy.int64),
                         op="Identity")
        elif isinstance(cst, float):
            Var.__init__(self, numpy.array([cst], dtype=numpy.float32),
                         op="Identity")
        else:
            raise NotImplementedError(
                f"Constant of type {type(cst)} are not implemented yet.")
        self._prefix = "cst"
