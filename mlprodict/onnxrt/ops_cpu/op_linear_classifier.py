# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
import numpy
from sklearn.utils.extmath import softmax
from ._op import OpRun


class LinearClassifier(OpRun):
    """
    Implements a linear classifier.
    """
    atts = {'classlabels_ints': [], 'classlabels_strings': [],
            'coefficients': None, 'intercepts': None,
            'multi_class': 0, 'post_transform': 'NONE'}

    def __init__(self, onnx_node, desc=None, **options):
        if desc is None:
            raise ValueError("desc should not be None.")
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=LinearClassifier.atts,
                       **options)
        if not isinstance(self.coefficients, numpy.ndarray):  # pylint: disable=E0203
            raise TypeError("coefficient must be an array not {}.".format(
                type(self.coefficients)))  # pylint: disable=E0203
        if len(getattr(self, "classlabels_ints", [])) == 0 and \
                len(getattr(self, 'classlabels_strings', [])) == 0:
            raise ValueError(
                "Fields classlabels_ints or classlabels_strings must be specified.")
        self.nb_class = max(len(getattr(self, 'classlabels_ints', [])),
                            len(getattr(self, 'classlabels_strings', [])))
        if self.nb_class == 2:
            self.nb_class = 1
        if len(self.coefficients.shape) != 1:  # pylint: disable=E0203
            raise ValueError("coefficient must be an array but has shape {}\n{}.".format(
                self.coefficients.shape, desc))  # pylint: disable=E0203
        n = self.coefficients.shape[0] // self.nb_class  # pylint: disable=E0203
        self.coefficients = self.coefficients.reshape(n, self.nb_class)

    def _run(self, x):  # pylint: disable=W0221
        score = numpy.dot(x, self.coefficients)
        if self.intercepts is not None:  # pylint: disable=E1101
            score += self.intercepts  # pylint: disable=E1101
        if self.post_transform == b'NONE':  # pylint: disable=E1101
            pass
        elif self.post_transform == b'LOGISTIC':  # pylint: disable=E1101
            score = softmax(score)
        else:
            raise NotImplementedError("Unknown post_transform: '{}'.".format(
                self.post_transform))  # pylint: disable=E1101

        if self.nb_class == 1:
            label = numpy.zeros((score.shape[0],))
            label[score > 0] = 1
        else:
            label = numpy.argmax(score, axis=1)
        return (label, score)
