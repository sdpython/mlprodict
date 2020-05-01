"""
.. _l-example-parallelism:

Benchmark Random Forests, Tree Ensemble
=======================================

The following scripts benchmarks different libraries
implementing random forest and boosting trees.

.. contents::
    :local:


Imports
+++++++
"""
import os
import pickle
import timeit
from pprint import pprint
import numpy
import pandas
import onnx
import onnxruntime
from onnxruntime import InferenceSession
from sklearn.datasets import make_classification
from skl2onnx import to_onnx
from mlprodict.onnx_conv import register_converters
from mlprodict.onnxrt.validate.validate_helper import measure_time
from mlprodict.onnxrt import OnnxInference

#############################
# Registers new converters for :epkg:`sklearn-onnx`.
register_converters()

#########################################
# Problem
# +++++++

max_depth = 7
n_classes = 10
n_estimators = 250
n_features = 200
REPEAT = 3
NUMBER = 1
train, test = 2000, 10000

print('dataset')
X_, y_ = make_classification(n_samples=train + test, n_features=n_features,
                             n_classes=n_classes, n_informative=n_classes // 2)
X_ = X_.astype(numpy.float32)
y_ = y_.astype(numpy.int64)
X_train, X_test = X_[:train], X_[train:]
y_train, y_test = y_[:train], y_[train:]

compilation = []


def train_cache(model, X_train, y_train, max_depth, n_estimators, n_classes):
    name = "cache-{}-N{}-f{}-d{}-e{}-cl{}.pkl".format(
        model.__class__.__name__, X_train.shape[0], X_train.shape[1],
        max_depth, n_estimators, n_classes)
    if os.path.exists(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        model.fit(X_train, y_train)
        with open(name, 'wb') as f:
            pickle.dump(model, f)
        return model


########################################
# RandomForestClassifier
# ++++++++++++++++++++++

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
print('train')
rf = train_cache(rf, X_train, y_train, max_depth, n_estimators, n_classes)

res = measure_time(rf.predict_proba, X_test[:10],
                   repeat=REPEAT, number=NUMBER,
                   div_by_number=True, first_run=True)
res['model'], res['runtime'] = rf.__class__.__name__, 'INNER'
pprint(res)

########################################
# ONNX


def measure_onnx_runtime(model, xt, repeat=REPEAT, number=NUMBER, verbose=True):
    if verbose:
        print(model.__class__.__name__)

    res = measure_time(model.predict_proba, xt,
                       repeat=repeat, number=number,
                       div_by_number=True, first_run=True)
    res['model'], res['runtime'] = model.__class__.__name__, 'INNER'
    res['N'] = X_test.shape[0]
    res["max_depth"] = max_depth
    res["n_estimators"] = n_estimators
    res["n_features"] = n_features
    if verbose:
        pprint(res)
    yield res

    onx = to_onnx(model, X_train[:1], options={id(model): {'zipmap': False}})

    oinf = OnnxInference(onx)
    res = measure_time(lambda x: oinf.run({'X': x}), xt,
                       repeat=repeat, number=number,
                       div_by_number=True, first_run=True)
    res['model'], res['runtime'] = model.__class__.__name__, 'NPY/C++'
    res['N'] = X_test.shape[0]
    res['size'] = len(onx.SerializeToString())
    res["max_depth"] = max_depth
    res["n_estimators"] = n_estimators
    res["n_features"] = n_features
    if verbose:
        pprint(res)
    yield res

    sess = InferenceSession(onx.SerializeToString())
    res = measure_time(lambda x: sess.run(None, {'X': x}), xt,
                       repeat=repeat, number=number,
                       div_by_number=True, first_run=True)
    res['model'], res['runtime'] = model.__class__.__name__, 'ORT'
    res['N'] = X_test.shape[0]
    res['size'] = len(onx.SerializeToString())
    res["max_depth"] = max_depth
    res["n_estimators"] = n_estimators
    res["n_features"] = n_features
    if verbose:
        pprint(res)
    yield res


compilation.extend(list(measure_onnx_runtime(rf, X_test)))


########################################
# HistGradientBoostingClassifier
# ++++++++++++++++++++++++++++++

from sklearn.ensemble import HistGradientBoostingClassifier
hist = HistGradientBoostingClassifier(
    max_iter=n_estimators, max_depth=max_depth)
print('train')
hist = train_cache(hist, X_train, y_train, max_depth, n_estimators, n_classes)

compilation.extend(list(measure_onnx_runtime(hist, X_test)))

########################################
# LightGBM
# ++++++++

from lightgbm import LGBMClassifier
lgb = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth)
print('train')
lgb = train_cache(lgb, X_train, y_train, max_depth, n_estimators, n_classes)

compilation.extend(list(measure_onnx_runtime(lgb, X_test)))

########################################
# XGBoost
# +++++++

from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
print('train')
xgb = train_cache(xgb, X_train, y_train, max_depth, n_estimators, n_classes)

compilation.extend(list(measure_onnx_runtime(xgb, X_test)))

##############################################
# Summary
# +++++++

df = pandas.DataFrame(compilation)
print(df)

piv = df.pivot("model", "runtime", "average")
print(piv)

piv.T.plot()
import matplotlib.pyplot as plt
plt.show()
