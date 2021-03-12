"""
@file
@brief Helpers to use numpy API to easily write converters
for :epkg:`scikit-learn` classes for :epkg:`onnx`.

.. versionadded:: 0.6
"""
from .onnx_numpy_wrapper import _created_classes_inst, wrapper_onnxnumpy_np


def update_registered_converter_npy(
        model, alias, convert_fct, shape_fct=None, overwrite=True,
        parser=None, options=None):
    """
    Registers or updates a converter for a new model so that
    it can be converted when inserted in a *scikit-learn* pipeline.
    This function assumes the converter is written with
    :ref:``.

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
    """
    from skl2onnx import update_registered_converter

    if hasattr(convert_fct, "compiled"):
        # type is wrapper_onnxnumpy
        sfct = convert_fct.compiled.convert_fct
    elif hasattr(convert_fct, 'signed_compiled'):
        # type is wrapper_onnxnumpy_np
        dfct = convert_fct.signed_compiled
    else:
        raise AttributeError(
            "Class %r must have attribute 'compiled' (object=%r)." % (
                type(convert_fct), convert_fct))
    update_registered_converter(
        model, alias, convert_fct=fct, shape_fct=shape_fct,
        overwrite=overwrite, parser=parser, options=options)


def onnxsklearn_transformer(op_version=None, runtime=None, signature=None,
                            register_class=None):
    """
    Decorator to declare a converter for a transformer implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: `'onnxruntime'` or one implemented by @see cl OnnxInference
    :param signature: if None, the signature is replace by a standard signature
        for transformer
    :param register_class: automatically register this converter
        for this class to :epkg:`sklearn-onnx`
    

    Equivalent to `onnxnumpy(arg)(foo)`.

    .. versionadded:: 0.6
    """
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
                register_class, "Sklearn%s" % getattr(register_class, "__name__", "noname"),
                res, shape_fct=default_shape_calculator, overwrite=False)            
        return res

    return decorator_fct
