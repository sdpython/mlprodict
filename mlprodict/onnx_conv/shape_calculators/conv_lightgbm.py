"""
@file
@brief Shape calculator for LGBMClassifier, handles doubles.
"""
import numbers
import numpy
from skl2onnx.common.data_types import (
    Int64TensorType, FloatTensorType, StringTensorType,
    DictionaryType, SequenceType, DoubleTensorType
)
from skl2onnx.common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_linear_classifier_output_shapes(operator):
    '''
    This operator maps an input feature vector into a scalar label if the number of outputs is one. If two outputs
    appear in this operator's output list, we should further generate a map storing all classes' probabilities.

    Allowed input/output patterns are
        1. [N, C] ---> [N, 1], A sequence of map

    Note that the second case is not allowed as long as ZipMap only produces dictionary.
    '''
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[
                                 FloatTensorType, Int64TensorType, DoubleTensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]
    real_type = operator.inputs[0].type.__class__

    class_labels = operator.raw_operator.classes_
    if all(isinstance(i, numpy.ndarray) for i in class_labels):
        class_labels = numpy.concatenate(class_labels)
    if all(isinstance(i, str) for i in class_labels):
        operator.outputs[0].type = StringTensorType(shape=[N])
        if len(class_labels) > 2 or operator.type != 'SklearnLinearSVC':
            # For multi-class classifier, we produce a map for encoding the probabilities of all classes
            if operator.target_opset < 7:
                operator.outputs[1].type = DictionaryType(
                    StringTensorType([1]), real_type([1]))
            else:
                operator.outputs[1].type = SequenceType(
                    DictionaryType(StringTensorType([]), real_type([])), N)
        else:
            # For binary LinearSVC, we produce probability of the positive class
            operator.outputs[1].type = real_type(shape=[N, 1])
    elif all(isinstance(i, (numbers.Real, bool, numpy.bool_)) for i in class_labels):
        operator.outputs[0].type = Int64TensorType(shape=[N])
        if len(class_labels) > 2 or operator.type != 'SklearnLinearSVC':
            # For multi-class classifier, we produce a map for encoding the probabilities of all classes
            if operator.target_opset < 7:
                operator.outputs[1].type = DictionaryType(
                    Int64TensorType([1]), real_type([1]))
            else:
                operator.outputs[1].type = SequenceType(
                    DictionaryType(Int64TensorType([]), real_type([])), N)
        else:
            # For binary LinearSVC, we produce probability of the positive class
            operator.outputs[1].type = real_type(shape=[N, 1])
    else:
        raise ValueError('Unsupported or mixed label types')
