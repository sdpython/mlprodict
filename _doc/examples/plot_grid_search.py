"""
.. _l-example-grid-search:

Grid search ONNX models
=======================

This example uses *OnnxTransformer* to freeze a model.
Many preprocessing are fitted, converted into :epkg:`ONNX`
and inserted into a pipeline with *OnnxTransformer*
si that they do not have to be fitted again.
The grid search will pick the best one for the task.

.. contents::
    :local:

Fit all preprocessings and serialize with ONNX
++++++++++++++++++++++++++++++++++++++++++++++
"""

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from mlprodict.sklapi import OnnxTransformer

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

dec_models = [
    PCA(n_components=1),
    PCA(n_components=2),
    StandardScaler(),
]

onx_bytes = []

for model in dec_models:
    model.fit(X_train)
    onx = convert_sklearn(
        model, initial_types=[('X', FloatTensorType((None, X.shape[1])))])
    onx_bytes.append(onx.SerializeToString())

##############################
# Pipeline with OnnxTransformer
# +++++++++++++++++++++++++++++++

pipe = make_pipeline(OnnxTransformer(onx_bytes[0]),
                     LogisticRegression(multi_class='ovr'))

################################
# Grid Search
# +++++++++++
#
# The serialized models are now used as a parameter
# in the grid search.

param_grid = [{'onnxtransformer__onnx_bytes': onx_bytes,
               'logisticregression__penalty': ['l2', 'l1'],
               'logisticregression__solver': ['liblinear', 'saga']
               }]


@ignore_warnings(category=ConvergenceWarning)
def fit(pipe, param_grid, cv=3):
    clf = GridSearchCV(pipe, param_grid, cv=3, n_jobs=1)
    clf.fit(X_train, y_train)
    return clf


clf = fit(pipe, param_grid)

y_true, y_pred = y_test, clf.predict(X_test)
cl = classification_report(y_true, y_pred)
print(cl)

#####################################
# Best preprocessing?
# +++++++++++++++++++
#
# We get the best parameters returned by the grid search
# and we search for it in the list of serialized
# preprocessing models.
# And the winner is...

bp = clf.best_params_
best_step = onx_bytes.index(bp["onnxtransformer__onnx_bytes"])
print(dec_models[best_step])
