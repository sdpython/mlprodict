# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
import inspect
import textwrap
from onnx.reference.ops import load_op as onnx_load_op
from onnx.defs import get_schema
from ..excs import MissingOperatorError
from ._op import OpRunCustom, OpFunction
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
    chosen_opset = opset or current_opset
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

    onnx_op = False
    if name_opset in _additional_ops:
        cl = _additional_ops[name_opset]
    elif name in _additional_ops:
        cl = _additional_ops[name]
    elif name_opset in d_op_list:
        cl = d_op_list[name_opset]
    elif name in d_op_list:
        cl = d_op_list[name]
    else:
        # finish
        try:
            cl = onnx_load_op(options.get('domain', ''),
                              name, opset)
        except ValueError as e:
            raise MissingOperatorError(
                f"Unable to load class for operator name={name}, "
                f"opset={opset}, options={options}, "
                f"_additional_ops={_additional_ops}.") from e
        onnx_op = True
        if cl is None:
            raise MissingOperatorError(  # pragma no cover
                "Operator '{}' from domain '{}' has no runtime yet. "
                "Available list:\n"
                "{} - {}".format(
                    name, onnx_node.domain,
                    "\n".join(sorted(_additional_ops)),
                    "\n".join(textwrap.wrap(
                        " ".join(
                            _ for _ in sorted(d_op_list)
                            if "_" not in _ and _ not in {
                                'cl', 'clo', 'name'})))))

        class _Wrapper:

            def _log(self, *args, **kwargs):
                pass

            @property
            def base_class(self):
                return self.__class__.__bases__[0]

            def _onnx_run(self, *args, **kwargs):
                cl = self.base_class
                new_kws = {}
                for k, v in kwargs.items():
                    if k not in {'attributes', 'verbose', 'fLOG'}:
                        new_kws[k] = v
                attributes = kwargs.get('attributes', None)
                if attributes is not None and len(attributes) > 0:
                    raise NotImplementedError(
                        f"attributes is not empty but not implemented yet, "
                        f"attribures={attributes}.")
                return cl.run(self, *args, **new_kws)  # pylint: disable=E1101

            def _onnx__run(self, *args, attributes=None, **kwargs):
                """
                Wraps ONNX call to OpRun._run.
                """
                cl = self.base_class
                if attributes is not None and len(attributes) > 0:
                    raise NotImplementedError(  # pragma: no cover
                        f"Linked attributes are not yet implemented for class "
                        f"{self.__class__!r}.")
                return cl._run(self, *args, **kwargs)  # pylint: disable=E1101

            def _onnx_need_context(self):
                cl = self.base_class
                return cl.need_context(self)  # pylint: disable=E1101

            def __init__(self, onnx_node, desc=None, **options):
                cl = self.base_class
                run_params = {'log': _Wrapper._log}
                cl.__init__(self, onnx_node, run_params)

        # wrapping the original class
        if inspect.isfunction(cl):
            domain = options.get('domain', '')
            if domain != '':
                raise TypeError(
                    f"Unable to create a class for operator {name!r} and "
                    f"opset {opset} based on {cl} of type={type(cl)}.")
            schema = get_schema(name, opset, domain)
            if schema.has_function:  # type: ignore
                from mlprodict.onnxrt import OnnxInference
                body = schema.function_body  # type: ignore
                sess = OnnxInference(body)
                new_cls = lambda *args, sess=sess: OpFunction(args[0], impl=sess)
            elif schema.has_context_dependent_function:  # type: ignore
                if node is None or input_types is None:
                    raise RuntimeContextError(
                        f"No registered implementation for operator {op_type!r} "
                        f"and domain {domain!r}, the operator has a context dependent function. "
                        f"but argument node or input_types is not defined.")
                from mlprodict.onnxrt import OnnxInference
                body = schema.get_context_dependent_function(
                    node.SerializeToString(),
                    [it.SerializeToString() for it in input_types])
                proto = FunctionProto()
                proto.ParseFromString(body)
                sess = ReferenceEvaluator(proto)
                new_cls = lambda *args, sess=sess: OpFunction(args[0], impl=sess)
            else:
                raise TypeError(
                    f"Unable to create a class for operator {name!r} and "
                    f"opset {opset} based on {cl} of type={type(cl)}.")
        else:
            try:
                new_cls = type(f"{name}_{opset}", (cl, ),
                               {'__init__': _Wrapper.__init__,
                                '_run': _Wrapper._onnx__run,
                                'run': _Wrapper._onnx_run,
                                'need_context': _Wrapper._onnx_need_context})
            except TypeError as e:
                raise TypeError(
                    f"Unable to create a class for operator {name!r} and "
                    f"opset {opset} based on {cl} of type={type(cl)}.") from e
        cl = new_cls

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
    if onnx_op:
        try:
            return cl(onnx_node, {'log': None})
        except TypeError as e:
            raise TypeError(  # pragma: no cover
                f"Unexpected issue with class {cl}.") from e
    try:
        return cl(onnx_node, desc=desc, runtime=runtime, **options)
    except TypeError as e:
        raise TypeError(  # pragma: no cover
            f"Unexpected issue with class {cl}.") from e
