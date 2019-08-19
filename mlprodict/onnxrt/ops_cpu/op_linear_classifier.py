# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from scipy.special import expit  # pylint: disable=E0611
from ._op import OpRunClassifierProb


class LinearClassifier(OpRunClassifierProb):

    atts = {'classlabels_ints': [], 'classlabels_strings': [],
            'coefficients': None, 'intercepts': None,
            'multi_class': 0, 'post_transform': b'NONE'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunClassifierProb.__init__(self, onnx_node, desc=desc,
                                     expected_attributes=LinearClassifier.atts,
                                     **options)
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
        if self.inplaces.get(0, False):
            numpy.dot(x, self.coefficients, out=x[:, 0])
            score = x[:, 0]
        else:
            score = numpy.dot(x, self.coefficients)
        if self.intercepts is not None:
            score += self.intercepts

        if self.post_transform == b'NONE':
            pass
        elif self.post_transform == b'LOGISTIC':
            expit(score, out=score)
        elif self.post_transform == b'SOFTMAX':
            numpy.subtract(score, score.max(axis=1)[
                           :, numpy.newaxis], out=score)
            numpy.exp(score, out=score)
            numpy.divide(score, score.sum(axis=1)[:, numpy.newaxis], out=score)
        else:
            raise NotImplementedError("Unknown post_transform: '{}'.".format(
                self.post_transform))

        if self.nb_class == 1:
            label = numpy.zeros((score.shape[0],), dtype=x.dtype)
            label[score > 0] = 1
        else:
            label = numpy.argmax(score, axis=1)
        return (label, score)
