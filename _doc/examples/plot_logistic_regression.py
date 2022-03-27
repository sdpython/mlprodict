# -*- coding: utf-8 -*-
"""
==========================================
Conversion of a logistic regression into C
==========================================

Simple example which shows how to predict with a logistic regression
using a code implemented in C. This configuration is significantly
faster in case of one-off prediction. It usually happens
when the machine learned model is embedded in a service.

"""

##############################
# Train a logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from mlprodict.grammar.grammar_sklearn import sklearn2graph

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
y[y == 2] = 1
lr = LogisticRegression()
lr.fit(X, y)

############################
# Conversion into a graph.
gr = sklearn2graph(lr, output_names=['Prediction', 'Score'])

######################################
# Conversion into C
ccode = gr.export(lang='c')
print(ccode['code'])

####################################
# This approach may work on small models.
# On bigger models with many dimensions,
# it would be better to use AVX instructions and parallelisation.
# Below, the optimisation this machine can offer.

from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
print(code_optimisation())
