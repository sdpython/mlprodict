"""
.. _l-example-tree-ensemble:

Benchmark Random Forests, Tree Ensemble
=======================================

The following script benchmarks different libraries
implementing random forests and boosting trees.
This benchmark can be replicated by installing the
following packages:

::

    python -m virtualenv env
    cd env
    pip install -i https://test.pypi.org/simple/ ort-nightly
    pip install git+https://github.com/microsoft/onnxconverter-common.git@jenkins
    pip install git+https://https://github.com/xadupre/sklearn-onnx.git@jenkins
    pip install mlprodict matplotlib scikit-learn pandas threadpoolctl
    pip install mlprodict lightgbm xgboost jinja2

.. contents::
    :local:

Import
++++++
"""
import os
import pickle
from pprint import pprint
import numpy
import pandas
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from onnxruntime import InferenceSession
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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
n_classes = 20
n_estimators = 500
n_features = 100
REPEAT = 3
NUMBER = 1
train, test = 1000, 10000

print('dataset')
X_, y_ = make_classification(n_samples=train + test, n_features=n_features,
                             n_classes=n_classes, n_informative=n_features - 3)
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
# ++++


def measure_onnx_runtime(model, xt, repeat=REPEAT, number=NUMBER,
                         verbose=True):
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

hist = HistGradientBoostingClassifier(
    max_iter=n_estimators, max_depth=max_depth)
print('train')
hist = train_cache(hist, X_train, y_train, max_depth, n_estimators, n_classes)

compilation.extend(list(measure_onnx_runtime(hist, X_test)))

########################################
# LightGBM
# ++++++++

lgb = LGBMClassifier(n_estimators=n_estimators,
                     max_depth=max_depth, pred_early_stop=False)
print('train')
lgb = train_cache(lgb, X_train, y_train, max_depth, n_estimators, n_classes)

compilation.extend(list(measure_onnx_runtime(lgb, X_test)))

########################################
# XGBoost
# +++++++

xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
print('train')
xgb = train_cache(xgb, X_train, y_train, max_depth, n_estimators, n_classes)

compilation.extend(list(measure_onnx_runtime(xgb, X_test)))

##############################################
# Summary
# +++++++
#
# All data
name = 'plot_time_tree_ensemble'
df = pandas.DataFrame(compilation)
df.to_csv('%s.csv' % name, index=False)
df.to_excel('%s.xlsx' % name, index=False)
df

#########################################
# Time per model and runtime.
piv = df.pivot("model", "runtime", "average")
piv

###########################################
# Graphs.
ax = piv.T.plot(kind="bar")
ax.set_title("Computation time ratio for %d observations and %d features\n"
             "lower is better for onnx runtimes" % X_test.shape)
plt.savefig('%s.png' % name)

###########################################
# Available optimisation on this machine:

from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
print(code_optimisation())

plt.show()
