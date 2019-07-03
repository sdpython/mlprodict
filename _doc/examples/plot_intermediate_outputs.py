# Licensed under the MIT License.

"""
.. _l-example-intermediate-outputs:

Investigate intermediate outupts
================================


.. contents::
    :local:

Train a model
+++++++++++++

A very basic example using
`TfidfVectorizer <https://scikit-learn.org/stable/modules/generated/
sklearn.feature_extraction.text.TfidfVectorizer.html>`_
on a dummy example.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


corpus = np.array([
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    ' ',
]).reshape((4, 1))
vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
vect.fit(corpus.ravel())
pred = vect.transform(corpus.ravel())

###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

from skl2onnx import convert_sklearn  # noqa
from skl2onnx.common.data_types import StringTensorType  # noqa

model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                             [('input', StringTensorType([1, 1]))])

print(model_onnx)
