# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
import textwrap
from ..excs import MissingOperatorError
from ._op import OpRunCustom
from ._op_list import __dict__ as d_op_list


_additional_ops = {}


def register_operator(cls, name=None, overwrite=True):
    """
    Registers a new runtime operator.

    @param      cls         class
    @param      name        by default ``cls.__name__``,
                            or *name* if defined
    @param      overwrite   overwrite or raise an exception
    """
    if name is None:
        name = cls.__name__
    if name not in _additional_ops:
        _additional_ops[name] = cls
    elif not overwrite:
        raise RuntimeError(  # pragma: no cover
            "Unable to overwrite existing operator '{}': {} "
            "by {}".format(name, _additional_ops[name], cls))


def load_op(onnx_node, desc=None, options=None, runtime=None):
    """
    Gets the operator related to the *onnx* node.

    :param onnx_node: :epkg:`onnx` node
    :param desc: internal representation
    :param options: runtime options
    :param runtime: runtime
    :param existing_functions: existing functions
    :return: runtime class
    """
    from ... import __max_supported_opset__
    if desc is None:
        raise ValueError("desc should not be None.")  # pragma no cover
    name = onnx_node.op_type
    opset = options.get('target_opset', None) if options is not None else None
    current_opset = __max_supported_opset__
    chosen_opset = current_opset
    if opset == current_opset:
        opset = None
    if opset is not None:
        if not isinstance(opset, int):
            raise TypeError(  # pragma no cover
                f"opset must be an integer not {type(opset)}")
        name_opset = name + "_" + str(opset)
        for op in range(opset, 0, -1):
            nop = name + "_" + str(op)
            if nop in d_op_list:
                name_opset = nop
                chosen_opset = op
                break
    else:
        name_opset = name

    if name_opset in _additional_ops:
        cl = _additional_ops[name_opset]
    elif name in _additional_ops:
        cl = _additional_ops[name]
    elif name_opset in d_op_list:
        cl = d_op_list[name_opset]
    elif name in d_op_list:
        cl = d_op_list[name]
    else:
        raise MissingOperatorError(  # pragma no cover
            "Operator '{}' from domain '{}' has no runtime yet. "
            "Available list:\n"
            "{} - {}".format(
                name, onnx_node.domain,
                "\n".join(sorted(_additional_ops)),
                "\n".join(textwrap.wrap(
                    " ".join(
                        _ for _ in sorted(d_op_list)
                        if "_" not in _ and _ not in {'cl', 'clo', 'name'})))))

    if hasattr(cl, 'version_higher_than'):
        opv = min(current_opset, chosen_opset)
        if cl.version_higher_than > opv:
            # The chosen implementation does not support
            # the opset version, we need to downgrade it.
            if ('target_opset' in options and
                    options['target_opset'] is not None):  # pragma: no cover
                raise RuntimeError(
                    "Supported version {} > {} (opset={}) required version, "
                    "unable to find an implementation version {} found "
                    "'{}'\n--ONNX--\n{}\n--AVAILABLE--\n{}".format(
                        cl.version_higher_than, opv, opset,
                        options['target_opset'], cl.__name__, onnx_node,
                        "\n".join(
                            _ for _ in sorted(d_op_list)
                            if "_" not in _ and _ not in {'cl', 'clo', 'name'})))
            options = options.copy()
            options['target_opset'] = current_opset
            return load_op(onnx_node, desc=desc, options=options)

    if options is None:
        options = {}  # pragma: no cover
    return cl(onnx_node, desc=desc, runtime=runtime, **options)
