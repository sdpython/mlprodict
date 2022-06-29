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
            "inputs should be OrderedDict not %r." % type(inputs))

    if node.op_type == 'Concat':
        type_set = set()
        for v in inputs.values():
            if not isinstance(v, Variable):
                raise TypeError(  # pragma: no cover
                    "Unexpected type %r for %r." % (type(v), v))
            type_set.add(v.dtype)
        if len(type_set) != 1:
            raise RuntimeError(
                "Unable to guess output type from %r (inputs=%r)."
                "" % (type_set, inputs))
        dtype = type_set.pop()
        if dtype is None:
            raise RuntimeError(
                "Guessed output type is None from inputs=%r." % (inputs, ))
        return dtype, [None, None]

    raise NotImplementedError(
        "Unable to infer type for node type %r and inputs=%r." % (
            node.op_type, inputs))
