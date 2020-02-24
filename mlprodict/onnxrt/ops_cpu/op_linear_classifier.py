# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from scipy.special import expit  # pylint: disable=E0611
from ._op import OpRunClassifierProb
from ._op_classifier_string import _ClassifierCommon
from ._op_numpy_helper import numpy_dot_inplace


class LinearClassifier(OpRunClassifierProb, _ClassifierCommon):

    atts = {'classlabels_ints': [], 'classlabels_strings': [],
            'coefficients': None, 'intercepts': None,
            'multi_class': 0, 'post_transform': b'NONE'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunClassifierProb.__init__(self, onnx_node, desc=desc,
                                     expected_attributes=LinearClassifier.atts,
                                     **options)
        self._post_process_label_attributes()
        if not isinstance(self.coefficients, numpy.ndarray):
            raise TypeError("coefficient must be an array not {}.".format(
                type(self.coefficients)))
        if len(getattr(self, "classlabels_ints", [])) == 0 and \
                len(getattr(self, 'classlabels_strings', [])) == 0:
            raise ValueError(
                "Fields classlabels_ints or classlabels_strings must be specified.")
        self.nb_class = max(len(getattr(self, 'classlabels_ints', [])),
                            len(getattr(self, 'classlabels_strings', [])))
        if len(self.coefficients.shape) != 1:
            raise ValueError("coefficient must be an array but has shape {}\n{}.".format(
                self.coefficients.shape, desc))
        n = self.coefficients.shape[0] // self.nb_class
        self.coefficients = self.coefficients.reshape(self.nb_class, n).T

    def _run(self, x):  # pylint: disable=W0221
        scores = numpy_dot_inplace(self.inplaces, x, self.coefficients)
        if self.intercepts is not None:
            scores += self.intercepts

        if self.post_transform == b'NONE':
            pass
        elif self.post_transform == b'LOGISTIC':
            expit(scores, out=scores)
        elif self.post_transform == b'SOFTMAX':
            numpy.subtract(scores, scores.max(axis=1)[
                           :, numpy.newaxis], out=scores)
            numpy.exp(scores, out=scores)
            numpy.divide(scores, scores.sum(axis=1)[
                         :, numpy.newaxis], out=scores)
        else:
            raise NotImplementedError("Unknown post_transform: '{}'.".format(
                self.post_transform))

        if self.nb_class == 1:
            label = numpy.zeros((scores.shape[0],), dtype=x.dtype)
            label[scores > 0] = 1
        else:
            label = numpy.argmax(scores, axis=1)
        return self._post_process_predicted_label(label, scores)
