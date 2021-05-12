"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
import copy
from onnx.helper import make_tensor
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.common._apply_operation import apply_concat, apply_identity
from ...onnx_tools.onnx2py_helper import _var_as_dict, guess_proto_dtype
from ...npy.onnx_version import FctVersion


def new_calculate_sklearn_function_transformer_output_shapes(operator):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support custom functions
    implemented with :ref:`l-numpy-onnxpy`.
    """
    fct = operator.raw_operator.func
    if hasattr(fct, 'signed_compiled'):
        dtype = guess_numpy_type(operator.inputs[0].type)
        fct = fct[FctVersion((dtype, ), None)]
    if hasattr(fct, 'compiled'):
        compiled = fct.compiled
        if not hasattr(compiled, 'onnx_'):
            raise RuntimeError(  # pragma: no cover
                "Attribute 'onnx_' is missing, function was not "
                "converted to onnx.")
        onx = compiled.onnx_
        graph = onx.graph
        outputs = graph.output

        # Let's assume there is only one output
        # with the same type as the input.
        # Only the shape changes.
        if len(outputs) != 1:
            raise RuntimeError(  # pragma: no cover
                "Only one output is allowed not %d." % len(outputs))
        input_type = operator.inputs[0].type.__class__
        if compiled.meta_.get('signature', None):
            dims = compiled.meta_['signature'].shape_calculator(
                operator.inputs[0].type.shape)
        else:
            N = operator.inputs[0].type.shape[0]
            dims = [N]
            out = outputs[0]
            try:
                extra_dims = out.type.tensor_type.shape.dim
            except AttributeError:  # pragma: no cover
                extra_dims = None
            if extra_dims is not None:
                dims.extend(extra_dims[1:])
        operator.outputs[0].type = input_type(dims)
        return

    if operator.raw_operator.func is not None:
        raise TypeError("FunctionTransformer is not supported unless the "
                        "transform function is of type %r "
                        "wrapped with onnxnumpy." % type(
                            operator.raw_operator.func))
    N = operator.inputs[0].type.shape[0]
    C = 0
    for variable in operator.inputs:
        if variable.type.shape[1] is not None:
            C += variable.type.shape[1]
        else:
            C = None
            break

    operator.outputs[0].type = operator.inputs[0].type.__class__([N, C])


def _copy_attributes(att):
    if hasattr(att, 'value'):
        return att.value
    vt = _var_as_dict(att)
    if vt['type']['kind'] == 'tensor':
        value = vt['value']
        return make_tensor(att.name, guess_proto_dtype(value.dtype),
                           value.shape, value.ravel().tolist())
    if vt['type']['kind'] == 'real':
        return vt['value']
    raise RuntimeError(  # pragma: no cover
        "Unable to copy attribute %r, got %r." % (att, vt))


def new_convert_sklearn_function_transformer(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support custom functions
    implemented with :ref:`l-numpy-onnxpy`.
    """
    op = operator.raw_operator
    fct = op.func
    if hasattr(fct, 'signed_compiled'):
        dtype = guess_numpy_type(operator.inputs[0].type)
        fct = fct[FctVersion((dtype, ), None)]
    if hasattr(fct, 'compiled'):
        compiled = fct.compiled
        if not hasattr(compiled, 'onnx_'):
            raise RuntimeError(  # pragma: no cover
                "Attribute 'onnx_' is missing, function was not "
                "converted to onnx.")
        onx = compiled.onnx_
        graph = onx.graph
        nodes = graph.node

        # renaming all intermediate variables
        names = []
        for node in nodes:
            for name in node.input:
                names.append(name)
            for name in node.output:
                names.append(name)
        names = set(names)
        names_mapping = {}
        for name in names:
            names_mapping[name] = scope.get_unique_variable_name(
                'ft_%s' % name)

        # adding identities
        apply_identity(scope, operator.inputs[0].full_name,
                       names_mapping[graph.input[0].name], container)
        apply_identity(scope, names_mapping[graph.output[0].name],
                       operator.outputs[0].full_name, container)

        # adding initializers
        for init in graph.initializer:
            init = copy.deepcopy(init)
            name = names_mapping[init.name]
            init.name = name
            content = init.SerializeToString()
            container.initializers_strings[content] = name
            container.initializers.append(init)

        # adding nodes
        for node in nodes:
            atts = {}
            for att in node.attribute:
                atts[att.name] = _copy_attributes(att)
            container.add_node(
                node.op_type,
                [names_mapping[n] for n in node.input],
                [names_mapping[n] for n in node.output],
                name=scope.get_unique_operator_name('ft_%s' % node.op_type),
                **atts)
        return

    if op.func is not None:
        raise TypeError(  # pragma: no cover
            "FunctionTransformer is not supported unless the "
            "transform function is of type %r or "
            "wrapped with onnxnumpy." % type(op.func))
    if len(operator.inputs) == 1:
        apply_identity(scope, operator.inputs[0].full_name,
                       operator.outputs[0].full_name, container)
    else:
        apply_concat(scope, [i.full_name for i in operator.inputs],
                     operator.outputs[0].full_name, container)
