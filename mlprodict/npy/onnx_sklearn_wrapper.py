"""
@file
@brief Helpers to use numpy API to easily write converters
for :epkg:`scikit-learn` classes for :epkg:`onnx`.

.. versionadded:: 0.6
"""
from skl2onnx import update_registered_converter
from skl2onnx.algebra.onnx_ops import OnnxIdentity  # pylint: disable=E0611
from .onnx_variable import OnnxVar
from .onnx_numpy_wrapper import _created_classes_inst, wrapper_onnxnumpy_np
from .onnx_numpy_annotation import NDArraySameType


def _shape_calculator_transformer(operator):
    """
    Default shape calculator for a transformer with one input
    and one output of the same type.

    .. versionadded:: 0.6
    """
    if not hasattr(operator, 'onnx_numpy_fct_'):
        raise AttributeError(
            "operator must have attribute 'onnx_numpy_fct_'.")
    X = operator.inputs
    if len(X) != 1:
        raise RuntimeError(
            "This function only supports one input not %r." % len(X))
    if len(operator.outputs) != 1:
        raise RuntimeError(
            "This function only supports one output not %r." % len(operator.outputs))
    cl = X[0].type.__class__
    dim = [X[0].type.shape[0], None]
    operator.outputs[0] = cl(dim)


def _converter_transformer(scope, operator, container):
    """
    Default converter for a transformer with one input
    and one output of the same type. It assumes instance *operator*
    has an attribute *onnx_numpy_fct_* from a function
    wrapped with decoarator :func:`onnxsklearn_transformer
    <mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_transformer>`.

    .. versionadded:: 0.6
    """
    if not hasattr(operator, 'onnx_numpy_fct_'):
        raise AttributeError(
            "operator must have attribute 'onnx_numpy_fct_'.")
    X = operator.inputs
    if len(X) != 1:
        raise RuntimeError(
            "This function only supports one input not %r." % len(X))
    if len(operator.outputs) != 1:
        raise RuntimeError(
            "This function only supports one output not %r." % len(operator.outputs))

    xvar = OnnxVar(X[0])
    fct_cl = operator.onnx_numpy_fct_

    opv = container.target_opset
    inst = fct_cl.fct(xvar, op=operator.raw_operator)
    onx = inst.to_algebra(op_version=opv)
    final = OnnxIdentity(onx, op_version=opv,
                         output_names=[operator.outputs[0].full_name])
    final.add_to(scope, container)


def update_registered_converter_npy(
        model, alias, convert_fct, shape_fct=None, overwrite=True,
        parser=None, options=None):
    """
    Registers or updates a converter for a new model so that
    it can be converted when inserted in a *scikit-learn* pipeline.
    This function assumes the converter is written as a function
    decoarated with :func:`onnxsklearn_transformer
    <mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_transformer>`.

    :param model: model class
    :param alias: alias used to register the model
    :param shape_fct: function which checks or modifies the expected
        outputs, this function should be fast so that the whole graph
        can be computed followed by the conversion of each model,
        parallelized or not
    :param convert_fct: function which converts a model
    :param overwrite: False to raise exception if a converter
        already exists
    :param parser: overwrites the parser as well if not empty
    :param options: registered options for this converter

    The alias is usually the library name followed by the model name.

    .. versionadded:: 0.6
    """
    if (hasattr(convert_fct, "compiled") or
            hasattr(convert_fct, 'signed_compiled')):
        # type is wrapper_onnxnumpy or wrapper_onnxnumpy_np
        obj = convert_fct
    else:
        raise AttributeError(
            "Class %r must have attribute 'compiled' or 'signed_compiled' "
            "(object=%r)." % (type(convert_fct), convert_fct))

    def addattr(operator, obj):
        operator.onnx_numpy_fct_ = obj
        return operator

    if shape_fct is None:
        local_shape_fct = (
            lambda operator:
            _shape_calculator_transformer(
                addattr(operator, obj)))
    else:
        local_shape_fct = shape_fct

    local_convert_fct = (
        lambda scope, operator, container:
        _converter_transformer(
            scope, addattr(operator, obj), container))

    update_registered_converter(
        model, alias, convert_fct=local_convert_fct,
        shape_fct=local_shape_fct, overwrite=overwrite,
        parser=parser, options=options)


def onnxsklearn_transformer(op_version=None, runtime=None, signature=None,
                            register_class=None):
    """
    Decorator to declare a converter for a transformer implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: `'onnxruntime'` or one implemented by @see cl OnnxInference
    :param signature: if None, the signature is replace by a standard signature
        for transformer ``NDArraySameType("all")``
    :param register_class: automatically register this converter
        for this class to :epkg:`sklearn-onnx`

    Equivalent to `onnxnumpy(arg)(foo)`.

    .. versionadded:: 0.6
    """
    if signature is None:
        signature = NDArraySameType("all")

    def default_shape_calculator(operator):
        op = operator.raw_operator
        if len(operator.inputs) != 1:
            raise NotImplementedError(
                "Default shape calculator only supports one input not %r (type=%r)"
                "." % (len(operator.inputs), type(op)))
        input_type = operator.inputs[0].type.__class__
        input_dim = operator.inputs[0].type.shape[0]
        output_type = input_type([input_dim, None])
        operator.outputs[0].type = output_type

    def decorator_fct(fct):
        name = "onnxsklearn_parser_%s_%s_%s" % (
            fct.__name__, str(op_version), runtime)
        newclass = type(
            name, (wrapper_onnxnumpy_np,), {
                '__doc__': fct.__doc__,
                '__getstate__': wrapper_onnxnumpy_np.__getstate__,
                '__setstate__': wrapper_onnxnumpy_np.__setstate__})
        _created_classes_inst.append(name, newclass)
        res = newclass(
            fct=fct, op_version=op_version, runtime=runtime,
            signature=signature)
        if register_class is not None:
            update_registered_converter_npy(
                register_class, "Sklearn%s" % getattr(
                    register_class, "__name__", "noname"),
                res, shape_fct=default_shape_calculator, overwrite=False)
        return res

    return decorator_fct
