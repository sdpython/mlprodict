"""
@file
@brief Helpers to use numpy API to easily write converters
for :epkg:`scikit-learn` classes for :epkg:`onnx`.

.. versionadded:: 0.6
"""
import logging
import numpy
from sklearn.base import (
    ClassifierMixin, ClusterMixin,
    RegressorMixin, TransformerMixin)
from .onnx_numpy_wrapper import _created_classes_inst, wrapper_onnxnumpy_np
from .onnx_numpy_annotation import NDArraySameType, NDArrayType
from .xop import OnnxOperatorTuple
from .xop_variable import Variable
from .xop import loadop
from ..plotting.text_plot import onnx_simple_text_plot


logger = logging.getLogger('xop')


def _skl2onnx_add_to_container(onx, scope, container, outputs):
    """
    Adds ONNX graph to :epkg:`skl2onnx` container and scope.

    :param onx: onnx graph
    :param scope: scope
    :param container: container
    """
    logger.debug("_skl2onnx_add_to_container:onx=%r outputs=%r",
                 type(onx), outputs)
    mapped_names = {x.name: x.name for x in onx.graph.input}
    opsets = {}
    for op in onx.opset_import:
        opsets[op.domain] = op.version

    # adding initializers
    for init in onx.graph.initializer:
        new_name = scope.get_unique_variable_name(init.name)
        mapped_names[init.name] = new_name
        container.add_initializer(new_name, None, None, init)

    # adding nodes
    for node in onx.graph.node:
        new_inputs = []
        for i in node.input:
            if i not in mapped_names:
                raise RuntimeError(  # pragma: no cover
                    f"Unable to find input {i!r} in {mapped_names!r}.")
            new_inputs.append(mapped_names[i])
        new_outputs = []
        for o in node.output:
            new_name = scope.get_unique_variable_name(o)
            mapped_names[o] = new_name
            new_outputs.append(new_name)

        atts = {}
        for att in node.attribute:
            if att.type == 1:  # .f
                value = att.f
            elif att.type == 2:  # .i
                value = att.i
            elif att.type == 3:  # .s
                value = att.s
            elif att.type == 4:  # .t
                value = att.t
            elif att.type == 6:  # .floats
                value = list(att.floats)
            elif att.type == 7:  # .ints
                value = list(att.ints)
            elif att.type == 8:  # .strings
                value = list(att.strings)
            else:
                raise NotImplementedError(  # pragma: no cover
                    f"Unable to copy attribute type {att.type!r} ({att!r}).")
            atts[att.name] = value

        container.add_node(
            node.op_type,
            name=scope.get_unique_operator_name('_sub_' + node.name),
            inputs=new_inputs, outputs=new_outputs, op_domain=node.domain,
            op_version=opsets.get(node.domain, None), **atts)

    # linking outputs
    if len(onx.graph.output) != len(outputs):
        raise RuntimeError(  # pragma: no cover
            "Output size mismatch %r != %r.\n--ONNX--\n%s" % (
                len(onx.graph.output), len(outputs),
                onnx_simple_text_plot(onx)))
    for out, var in zip(onx.graph.output, outputs):
        container.add_node(
            'Identity', name=scope.get_unique_operator_name(
                '_sub_' + out.name),
            inputs=[mapped_names[out.name]], outputs=[var.onnx_name])


def _common_shape_calculator_t(operator):
    if not hasattr(operator, 'onnx_numpy_fct_'):
        raise AttributeError(
            "operator must have attribute 'onnx_numpy_fct_'.")
    X = operator.inputs
    if len(X) != 1:
        raise RuntimeError(
            f"This function only supports one input not {len(X)!r}.")
    if len(operator.outputs) != 1:
        raise RuntimeError(
            f"This function only supports one output not {len(operator.outputs)!r}.")
    op = operator.raw_operator
    cl = X[0].type.__class__
    dim = [X[0].type.shape[0], getattr(op, 'n_outputs_', None)]
    operator.outputs[0].type = cl(dim)


def _shape_calculator_transformer(operator):
    """
    Default shape calculator for a transformer with one input
    and one output of the same type.

    .. versionadded:: 0.6
    """
    _common_shape_calculator_t(operator)


def _shape_calculator_regressor(operator):
    """
    Default shape calculator for a regressor with one input
    and one output of the same type.

    .. versionadded:: 0.6
    """
    _common_shape_calculator_t(operator)


def _common_shape_calculator_int_t(operator):
    if not hasattr(operator, 'onnx_numpy_fct_'):
        raise AttributeError(
            "operator must have attribute 'onnx_numpy_fct_'.")
    X = operator.inputs
    if len(X) != 1:
        raise RuntimeError(
            f"This function only supports one input not {len(X)!r}.")
    if len(operator.outputs) != 2:
        raise RuntimeError(
            f"This function only supports two outputs not {len(operator.outputs)!r}.")
    from skl2onnx.common.data_types import Int64TensorType  # delayed
    op = operator.raw_operator
    cl = X[0].type.__class__
    dim = [X[0].type.shape[0], getattr(op, 'n_outputs_', None)]
    operator.outputs[0].type = Int64TensorType(dim[:1])
    operator.outputs[1].type = cl(dim)


def _shape_calculator_classifier(operator):
    """
    Default shape calculator for a classifier with one input
    and two outputs, label (int64) and probabilites of the same type.

    .. versionadded:: 0.6
    """
    _common_shape_calculator_int_t(operator)


def _shape_calculator_cluster(operator):
    """
    Default shape calculator for a clustering with one input
    and two outputs, label (int64) and distances of the same type.

    .. versionadded:: 0.6
    """
    _common_shape_calculator_int_t(operator)


def _common_converter_begin(scope, operator, container, n_outputs):
    if not hasattr(operator, 'onnx_numpy_fct_'):
        raise AttributeError(
            "operator must have attribute 'onnx_numpy_fct_'.")
    X = operator.inputs
    if len(X) != 1:
        raise RuntimeError(
            f"This function only supports one input not {len(X)!r}.")
    if len(operator.outputs) != n_outputs:
        raise RuntimeError(
            "This function only supports %d output not %r." % (
                n_outputs, len(operator.outputs)))

    # First conversion of the model to onnx
    # Then addition of the onnx graph to the main graph.
    from .onnx_variable import OnnxVar
    new_var = Variable.from_skl2onnx(X[0])
    xvar = OnnxVar(new_var)
    fct_cl = operator.onnx_numpy_fct_

    opv = container.target_opset
    logger.debug("_common_converter_begin:xvar=%r op=%s",
                 xvar, type(operator.raw_operator))
    inst = fct_cl.fct(xvar, op_=operator.raw_operator)
    logger.debug("_common_converter_begin:inst=%r opv=%r fct_cl.fct=%r",
                 type(inst), opv, fct_cl.fct)
    onx = inst.to_algebra(op_version=opv)
    logger.debug("_common_converter_begin:end:onx=%r", type(onx))
    return new_var, onx


def _common_converter_t(scope, operator, container):
    logger.debug("_common_converter_t:op=%r -> %r",
                 operator.inputs, operator.outputs)
    OnnxIdentity = loadop('Identity')
    opv = container.target_opset
    new_var, onx = _common_converter_begin(scope, operator, container, 1)
    final = OnnxIdentity(onx, op_version=opv,
                         output_names=[operator.outputs[0].full_name])
    onx_model = final.to_onnx(
        [new_var], [Variable.from_skl2onnx(o) for o in operator.outputs],
        target_opset=opv)
    _skl2onnx_add_to_container(onx_model, scope, container, operator.outputs)
    logger.debug("_common_converter_t:end")


def _converter_transformer(scope, operator, container):
    """
    Default converter for a transformer with one input
    and one output of the same type. It assumes instance *operator*
    has an attribute *onnx_numpy_fct_* from a function
    wrapped with decorator :func:`onnxsklearn_transformer
    <mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_transformer>`.

    .. versionadded:: 0.6
    """
    _common_converter_t(scope, operator, container)


def _converter_regressor(scope, operator, container):
    """
    Default converter for a regressor with one input
    and one output of the same type. It assumes instance *operator*
    has an attribute *onnx_numpy_fct_* from a function
    wrapped with decorator :func:`onnxsklearn_regressor
    <mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_regressor>`.

    .. versionadded:: 0.6
    """
    _common_converter_t(scope, operator, container)


def _common_converter_int_t(scope, operator, container):
    logger.debug("_common_converter_int_t:op=%r -> %r",
                 operator.inputs, operator.outputs)
    OnnxIdentity = loadop('Identity')
    opv = container.target_opset
    new_var, onx = _common_converter_begin(scope, operator, container, 2)

    if isinstance(onx, OnnxOperatorTuple):
        if len(operator.outputs) != len(onx):
            raise RuntimeError(  # pragma: no cover
                "Mismatched number of outputs expected %d, got %d." % (
                    len(operator.outputs), len(onx)))
        first_output = None
        other_outputs = []
        for out, ox in zip(operator.outputs, onx):
            if not hasattr(ox, 'add_to'):
                raise TypeError(  # pragma: no cover
                    "Unexpected type for onnx graph %r, inst=%r." % (
                        type(ox), type(operator.raw_operator)))
            final = OnnxIdentity(ox, op_version=opv,
                                 output_names=[out.full_name])
            if first_output is None:
                first_output = final
            else:
                other_outputs.append(final)

        onx_model = first_output.to_onnx(
            [new_var],
            [Variable.from_skl2onnx(o) for o in operator.outputs],
            target_opset=opv, other_outputs=other_outputs)
        _skl2onnx_add_to_container(
            onx_model, scope, container, operator.outputs)
        logger.debug("_common_converter_int_t:1:end")
    else:
        final = OnnxIdentity(onx, op_version=opv,
                             output_names=[operator.outputs[0].full_name])
        onx_model = final.to_onnx(
            [new_var],
            [Variable.from_skl2onnx(o) for o in operator.outputs],
            target_opset=opv)
        _skl2onnx_add_to_container(
            onx_model, scope, container, operator.outputs)
        logger.debug("_common_converter_int_t:2:end")


def _converter_classifier(scope, operator, container):
    """
    Default converter for a classifier with one input
    and two outputs, label and probabilities of the same input type.
    It assumes instance *operator*
    has an attribute *onnx_numpy_fct_* from a function
    wrapped with decorator :func:`onnxsklearn_classifier
    <mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_classifier>`.

    .. versionadded:: 0.6
    """
    _common_converter_int_t(scope, operator, container)


def _converter_cluster(scope, operator, container):
    """
    Default converter for a clustering with one input
    and two outputs, label and distances of the same input type.
    It assumes instance *operator*
    has an attribute *onnx_numpy_fct_* from a function
    wrapped with decorator :func:`onnxsklearn_cluster
    <mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_cluster>`.

    .. versionadded:: 0.6
    """
    _common_converter_int_t(scope, operator, container)


_default_cvt = {
    ClassifierMixin: (_shape_calculator_classifier, _converter_classifier),
    ClusterMixin: (_shape_calculator_cluster, _converter_cluster),
    RegressorMixin: (_shape_calculator_regressor, _converter_regressor),
    TransformerMixin: (_shape_calculator_transformer, _converter_transformer),
}


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
        raise AttributeError(  # pragma: no cover
            "Class %r must have attribute 'compiled' or 'signed_compiled' "
            "(object=%r)." % (type(convert_fct), convert_fct))

    def addattr(operator, obj):
        operator.onnx_numpy_fct_ = obj
        return operator

    if issubclass(model, TransformerMixin):
        defcl = TransformerMixin
    elif issubclass(model, RegressorMixin):
        defcl = RegressorMixin
    elif issubclass(model, ClassifierMixin):
        defcl = ClassifierMixin
    elif issubclass(model, ClusterMixin):
        defcl = ClusterMixin
    else:
        defcl = None

    if shape_fct is not None:
        raise NotImplementedError(  # pragma: no cover
            "Custom shape calculator are not implemented yet.")

    shc = _default_cvt[defcl][0]
    local_shape_fct = (
        lambda operator: shc(addattr(operator, obj)))

    cvtc = _default_cvt[defcl][1]
    local_convert_fct = (
        lambda scope, operator, container:
        cvtc(scope, addattr(operator, obj), container))

    from skl2onnx import update_registered_converter  # delayed
    update_registered_converter(
        model, alias, convert_fct=local_convert_fct,
        shape_fct=local_shape_fct, overwrite=overwrite,
        parser=parser, options=options)


def _internal_decorator(fct, op_version=None, runtime=None, signature=None,
                        register_class=None, overwrite=True, options=None):
    name = f"onnxsklearn_parser_{fct.__name__}_{str(op_version)}_{runtime}"
    newclass = type(
        name, (wrapper_onnxnumpy_np,), {
            '__doc__': fct.__doc__,
            '__name__': name,
            '__getstate__': wrapper_onnxnumpy_np.__getstate__,
            '__setstate__': wrapper_onnxnumpy_np.__setstate__})
    _created_classes_inst.append(name, newclass)
    res = newclass(
        fct=fct, op_version=op_version, runtime=runtime,
        signature=signature)
    if register_class is not None:
        update_registered_converter_npy(
            register_class, f"Sklearn{getattr(register_class, '__name__', 'noname')}",
            res, shape_fct=None, overwrite=overwrite, options=options)
    return res


def onnxsklearn_transformer(op_version=None, runtime=None, signature=None,
                            register_class=None, overwrite=True):
    """
    Decorator to declare a converter for a transformer implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: `'onnxruntime'` or one implemented by @see cl OnnxInference
    :param signature: if None, the signature is replaced by a standard signature
        for transformer ``NDArraySameType("all")``
    :param register_class: automatically register this converter
        for this class to :epkg:`sklearn-onnx`
    :param overwrite: overwrite existing registered function if any

    .. versionadded:: 0.6
    """
    if signature is None:
        signature = NDArraySameType("all")

    def decorator_fct(fct):
        return _internal_decorator(fct, signature=signature,
                                   op_version=op_version,
                                   runtime=runtime,
                                   register_class=register_class,
                                   overwrite=overwrite)
    return decorator_fct


def onnxsklearn_regressor(op_version=None, runtime=None, signature=None,
                          register_class=None, overwrite=True):
    """
    Decorator to declare a converter for a regressor implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: `'onnxruntime'` or one implemented by @see cl OnnxInference
    :param signature: if None, the signature is replaced by a standard signature
        for transformer ``NDArraySameType("all")``
    :param register_class: automatically register this converter
        for this class to :epkg:`sklearn-onnx`
    :param overwrite: overwrite existing registered function if any

    .. versionadded:: 0.6
    """
    if signature is None:
        signature = NDArraySameType("all")

    def decorator_fct(fct):
        return _internal_decorator(fct, signature=signature,
                                   op_version=op_version,
                                   runtime=runtime,
                                   register_class=register_class,
                                   overwrite=overwrite)
    return decorator_fct


def onnxsklearn_classifier(op_version=None, runtime=None, signature=None,
                           register_class=None, overwrite=True):
    """
    Decorator to declare a converter for a classifier implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: `'onnxruntime'` or one implemented by @see cl OnnxInference
    :param signature: if None, the signature is replaced by a standard signature
        for transformer ``NDArraySameType("all")``
    :param register_class: automatically register this converter
        for this class to :epkg:`sklearn-onnx`
    :param overwrite: overwrite existing registered function if any

    .. versionadded:: 0.6
    """
    if signature is None:
        signature = NDArrayType(("T:all", ), dtypes_out=((numpy.int64, ), 'T'))

    def decorator_fct(fct):
        return _internal_decorator(fct, signature=signature,
                                   op_version=op_version,
                                   runtime=runtime,
                                   register_class=register_class,
                                   overwrite=overwrite,
                                   options={'zipmap': [False, True, 'columns'],
                                            'nocl': [False, True]})
    return decorator_fct


def onnxsklearn_cluster(op_version=None, runtime=None, signature=None,
                        register_class=None, overwrite=True):
    """
    Decorator to declare a converter for a cluster implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: `'onnxruntime'` or one implemented by @see cl OnnxInference
    :param signature: if None, the signature is replaced by a standard signature
        for transformer ``NDArraySameType("all")``
    :param register_class: automatically register this converter
        for this class to :epkg:`sklearn-onnx`
    :param overwrite: overwrite existing registered function if any

    .. versionadded:: 0.6
    """
    if signature is None:
        signature = NDArrayType(("T:all", ), dtypes_out=((numpy.int64, ), 'T'))

    def decorator_fct(fct):
        return _internal_decorator(fct, signature=signature,
                                   op_version=op_version,
                                   runtime=runtime,
                                   register_class=register_class,
                                   overwrite=overwrite)
    return decorator_fct


def _call_validate(self, X):
    if hasattr(self, "_validate_onnx_data"):
        return self._validate_onnx_data(X)
    return X


def _internal_method_decorator(register_class, method, op_version=None,
                               runtime=None, signature=None,
                               method_names=None, overwrite=True,
                               options=None):
    if isinstance(method_names, str):
        method_names = (method_names, )

    if issubclass(register_class, TransformerMixin):
        if signature is None:
            signature = NDArraySameType("all")
        if method_names is None:
            method_names = ("transform", )
    elif issubclass(register_class, RegressorMixin):
        if signature is None:
            signature = NDArraySameType("all")
        if method_names is None:
            method_names = ("predict", )
    elif issubclass(register_class, ClassifierMixin):
        if signature is None:
            signature = NDArrayType(
                ("T:all", ), dtypes_out=((numpy.int64, ), 'T'))
        if method_names is None:
            method_names = ("predict", "predict_proba")
        if options is None:
            options = {'zipmap': [False, True, 'columns'],
                       'nocl': [False, True]}
    elif issubclass(register_class, ClusterMixin):
        if signature is None:
            signature = NDArrayType(
                ("T:all", ), dtypes_out=((numpy.int64, ), 'T'))
        if method_names is None:
            method_names = ("predict", "transform")
    elif method_names is None:  # pragma: no cover
        raise RuntimeError(
            "No obvious API was detected (one among %s), "
            "then 'method_names' must be specified and not left "
            "empty." % (", ".join(map(lambda s: s.__name__, _default_cvt))))

    if method_names is None:
        raise RuntimeError(  # pragma: no cover
            "Methods to overwrite are not known for class %r and "
            "method %r." % (register_class, method))
    if signature is None:
        raise RuntimeError(  # pragma: no cover
            "Methods to overwrite are not known for class %r and "
            "method %r." % (register_class, method))

    name = f"onnxsklearn_parser_{register_class.__name__}_{str(op_version)}_{runtime}"
    newclass = type(
        name, (wrapper_onnxnumpy_np,), {
            '__doc__': method.__doc__,
            '__name__': name,
            '__getstate__': wrapper_onnxnumpy_np.__getstate__,
            '__setstate__': wrapper_onnxnumpy_np.__setstate__})
    _created_classes_inst.append(name, newclass)

    def _check_(op):
        if isinstance(op, str):
            raise TypeError(  # pragma: no cover
                f"Unexpected type: {type(op)!r}: {op!r}.")
        return op

    res = newclass(
        fct=lambda *args, op_=None, **kwargs: method(
            _check_(op_), *args, **kwargs),
        op_version=op_version, runtime=runtime, signature=signature,
        fctsig=method)

    if len(method_names) == 1:
        name = method_names[0]
        if hasattr(register_class, name):
            raise RuntimeError(  # pragma: no cover
                "Cannot overwrite method %r because it already exists in "
                "class %r." % (name, register_class))
        m = lambda self, X: res(_call_validate(self, X), op_=self)
        setattr(register_class, name, m)
    elif len(method_names) == 0:
        raise RuntimeError("No available method.")  # pragma: no cover
    else:
        m = lambda self, X: res(_call_validate(self, X), op_=self)
        setattr(register_class, method.__name__ + "_", m)
        for iname, name in enumerate(method_names):
            if hasattr(register_class, name):
                raise RuntimeError(  # pragma: no cover
                    "Cannot overwrite method %r because it already exists in "
                    "class %r." % (name, register_class))
            m = (lambda self, X, index_output=iname:
                 res(_call_validate(self, X), op_=self)[index_output])
            setattr(register_class, name, m)

    update_registered_converter_npy(
        register_class, f"Sklearn{getattr(register_class, '__name__', 'noname')}",
        res, shape_fct=None, overwrite=overwrite,
        options=options)
    return res


def onnxsklearn_class(method_name, op_version=None, runtime=None,
                      signature=None, method_names=None,
                      overwrite=True):
    """
    Decorator to declare a converter for a class derivated from
    :epkg:`scikit-learn`, implementing inference method
    and using :epkg:`numpy` syntax but executed with
    :epkg:`ONNX` operators.

    :param method_name: name of the method implementing the
        inference method with :epkg:`numpy` API for ONNX
    :param op_version: :epkg:`ONNX` opset version
    :param runtime: `'onnxruntime'` or one implemented by @see cl OnnxInference
    :param signature: if None, the signature is replaced by a standard signature
        depending on the model kind, otherwise, it is the signature of the
        ONNX function
    :param method_names: if None, method names is guessed based on
        the class kind (transformer, regressor, classifier, clusterer)
    :param overwrite: overwrite existing registered function if any

    .. versionadded:: 0.6
    """
    def decorator_class(objclass):
        _internal_method_decorator(
            objclass, method=getattr(objclass, method_name),
            signature=signature, op_version=op_version,
            runtime=runtime, method_names=method_names,
            overwrite=overwrite)
        return objclass

    return decorator_class
