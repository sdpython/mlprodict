"""
@file
@brief Helpers to use numpy API to easily write converters
for :epkg:`scikit-learn` classes for :epkg:`onnx`.

.. versionadded:: 0.6
"""
import numpy
from sklearn.base import (
    ClassifierMixin, ClusterMixin,
    RegressorMixin, TransformerMixin)
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx.algebra.onnx_ops import OnnxIdentity  # pylint: disable=E0611
from .onnx_variable import OnnxVar, TupleOnnxAny
from .onnx_numpy_wrapper import _created_classes_inst, wrapper_onnxnumpy_np
from .onnx_numpy_annotation import NDArraySameType, NDArrayType


def _common_shape_calculator_t(operator):
    if not hasattr(operator, 'onnx_numpy_fct_'):
        raise AttributeError(
            "operator must have attribute 'onnx_numpy_fct_'.")
    X = operator.inputs
    if len(X) != 1:
        raise RuntimeError(
            "This function only supports one input not %r." % len(X))
    if len(operator.outputs) != 1:
        raise RuntimeError(
            "This function only supports one output not %r." % len(
                operator.outputs))
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
            "This function only supports one input not %r." % len(X))
    if len(operator.outputs) != 2:
        raise RuntimeError(
            "This function only supports two outputs not %r." % len(
                operator.outputs))
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


def _common_converter_t(scope, operator, container):
    if not hasattr(operator, 'onnx_numpy_fct_'):
        raise AttributeError(
            "operator must have attribute 'onnx_numpy_fct_'.")
    X = operator.inputs
    if len(X) != 1:
        raise RuntimeError(
            "This function only supports one input not %r." % len(X))
    if len(operator.outputs) != 1:
        raise RuntimeError(
            "This function only supports one output not %r." % len(
                operator.outputs))

    xvar = OnnxVar(X[0])
    fct_cl = operator.onnx_numpy_fct_

    opv = container.target_opset
    inst = fct_cl.fct(xvar, op_=operator.raw_operator)
    onx = inst.to_algebra(op_version=opv)
    final = OnnxIdentity(onx, op_version=opv,
                         output_names=[operator.outputs[0].full_name])
    final.add_to(scope, container)


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
    if not hasattr(operator, 'onnx_numpy_fct_'):
        raise AttributeError(
            "operator must have attribute 'onnx_numpy_fct_'.")
    X = operator.inputs
    if len(X) != 1:
        raise RuntimeError(
            "This function only supports one input not %r." % len(X))
    if len(operator.outputs) != 2:
        raise RuntimeError(
            "This function only supports two outputs not %r." % len(
                operator.outputs))

    xvar = OnnxVar(X[0])
    fct_cl = operator.onnx_numpy_fct_

    opv = container.target_opset
    inst = fct_cl.fct(xvar, op_=operator.raw_operator)
    onx = inst.to_algebra(op_version=opv)
    if isinstance(onx, TupleOnnxAny):
        if len(operator.outputs) != len(onx):
            raise RuntimeError(  # pragma: no cover
                "Mismatched number of outputs expected %d, got %d." % (
                    len(operator.outputs), len(onx)))
        for out, ox in zip(operator.outputs, onx):
            if not hasattr(ox, 'add_to'):
                raise TypeError(  # pragma: no cover
                    "Unexpected type for onnx graph %r, inst=%r." % (
                        type(ox), type(inst)))
            final = OnnxIdentity(ox, op_version=opv,
                                 output_names=[out.full_name])
            final.add_to(scope, container)
    else:
        final = OnnxIdentity(onx, op_version=opv,
                             output_names=[operator.outputs[0].full_name])
        final.add_to(scope, container)


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

    update_registered_converter(
        model, alias, convert_fct=local_convert_fct,
        shape_fct=local_shape_fct, overwrite=overwrite,
        parser=parser, options=options)


def _internal_decorator(fct, op_version=None, runtime=None, signature=None,
                        register_class=None, overwrite=True, options=None):
    name = "onnxsklearn_parser_%s_%s_%s" % (
        fct.__name__, str(op_version), runtime)
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
            register_class, "Sklearn%s" % getattr(
                register_class, "__name__", "noname"),
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
    elif method_names is None:
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

    name = "onnxsklearn_parser_%s_%s_%s" % (
        register_class.__name__, str(op_version), runtime)
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
                "Unexpected type: %r: %r." % (type(op), op))
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
        register_class, "Sklearn%s" % getattr(
            register_class, "__name__", "noname"),
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
