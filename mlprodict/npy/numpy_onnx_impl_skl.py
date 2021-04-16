"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.

.. versionadded:: 0.6
"""
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from .onnx_variable import MultiOnnxVar, OnnxVar


def linear_regression(x, *, model=None):
    """
    Returns a linear regression converted into ONNX.

    :param x: array, variable name, instance of :class:`OnnxVar
        <mlprodict.npy.onnx_variable.OnnxVar>`
    :param model: instance of :epkg:`sklearn:linear_model:LinearRegression`
    :return: instance of :class:`OnnxVar
        <mlprodict.npy.onnx_variable.OnnxVar>`
    """
    return OnnxVar(model, x, op=OnnxSubEstimator)


def logistic_regression(x, *, model=None):
    """
    Returns a logistic regression converted into ONNX,
    option *zipmap* is set to false.

    :param x: array, variable name, instance of :class:`OnnxVar
        <mlprodict.npy.onnx_variable.OnnxVar>`
    :param model: instance of :epkg:`sklearn:linear_model:LinearRegression`
    :return: instance of :class:`MultiOnnxVar
        <mlprodict.npy.onnx_variable.MultiOnnxVar>`, first
        output is labels, second one is the probabilities
    """
    return MultiOnnxVar(model, x, op=OnnxSubEstimator,
                        options={'zipmap': False})


def classifier(x, *, model=None):
    """
    Returns any classifier from :epkg:`scikit-learn`
    converted into ONNX assuming a converter is registered
    with :epkg:`sklearn-onnx`. Option *zipmap* is set to false.

    :param x: array, variable name, instance of :class:`OnnxVar
        <mlprodict.npy.onnx_variable.OnnxVar>`
    :param model: instance of a classifier
    :return: instance of :class:`MultiOnnxVar
        <mlprodict.npy.onnx_variable.MultiOnnxVar>`, first
        output is labels, second one is the probabilities
    """
    return MultiOnnxVar(model, x, op=OnnxSubEstimator,
                        options={'zipmap': False})


def cluster(x, *, model=None):
    """
    Returns any cluster from :epkg:`scikit-learn`
    converted into ONNX assuming a converter is registered
    with :epkg:`sklearn-onnx`. Option *zipmap* is set to false.

    :param x: array, variable name, instance of :class:`OnnxVar
        <mlprodict.npy.onnx_variable.OnnxVar>`
    :param model: instance of a cluster
    :return: instance of :class:`MultiOnnxVar
        <mlprodict.npy.onnx_variable.MultiOnnxVar>`, first
        output is labels, second one is the probabilities
    """
    return MultiOnnxVar(model, x, op=OnnxSubEstimator)


def regressor(x, *, model=None):
    """
    Returns any regressor from :epkg:`scikit-learn`
    converted into ONNX assuming a converter is registered
    with :epkg:`sklearn-onnx`.

    :param x: array, variable name, instance of :class:`OnnxVar
        <mlprodict.npy.onnx_variable.OnnxVar>`
    :param model: instance of a regressor
    :return: instance of :class:`OnnxVar
        <mlprodict.npy.onnx_variable.OnnxVar>`
    """
    return OnnxVar(model, x, op=OnnxSubEstimator)


def transformer(x, *, model=None):
    """
    Returns any transformer from :epkg:`scikit-learn`
    converted into ONNX assuming a converter is registered
    with :epkg:`sklearn-onnx`.

    :param x: array, variable name, instance of :class:`OnnxVar
        <mlprodict.npy.onnx_variable.OnnxVar>`
    :param model: instance of a transformer
    :return: instance of :class:`OnnxVar
        <mlprodict.npy.onnx_variable.OnnxVar>`
    """
    return OnnxVar(model, x, op=OnnxSubEstimator)
