# pylint: disable=E0602
"""
@file
@brief Xop helpers.

.. versionadded:: 0.9
"""
from .xop_variable import Variable


def _infer_node_output(node, inputs):
    """
    Infers node outputs for a specific type.

    :param node: :epkg:`NodeProto`
    :param outputs: known inputs
    :return: dtype
    """
    if not isinstance(inputs, dict):
        raise TypeError(  # pragma: no cover
            f"inputs should be OrderedDict not {type(inputs)!r}.")

    if node.op_type == 'Concat':
        type_set = set()
        for v in inputs.values():
            if not isinstance(v, Variable):
                raise TypeError(  # pragma: no cover
                    f"Unexpected type {type(v)!r} for {v!r}.")
            type_set.add(v.dtype)
        if len(type_set) != 1:
            raise RuntimeError(  # pragma: no cover
                f"Unable to guess output type from {type_set!r} (inputs={inputs!r}).")
        dtype = type_set.pop()
        if dtype is None:
            raise RuntimeError(  # pragma: no cover
                f"Guessed output type is None from inputs={inputs!r}.")
        return dtype, [None, None]

    raise NotImplementedError(  # pragma: no cover
        f"Unable to infer type for node type {node.op_type!r} and inputs={inputs!r}.")
