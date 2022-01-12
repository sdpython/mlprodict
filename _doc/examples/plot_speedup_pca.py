"""
.. _l-Speedup-pca:

Speed up scikit-learn inference with ONNX
=========================================

Is it possible to make :epkg:`scikit-learn` faster with ONNX?
That's question this example tries to answer. The scenario is
is the following:

* a model is trained
* it is converted into ONNX for inference
* it selects a runtime to compute the prediction

The following runtime are tested:

* `python`: python runtime for ONNX
* `onnxruntime1`: :epkg:`onnxruntime`
* `numpy`: the ONNX graph is converted into numpy code
* `numba`: the numpy code is accelerated with :epkg:`numba`.

.. contents::
    :local:

PCA
+++

Let's look at a very simple model, a PCA.
"""

import numpy
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from pyquickhelper.pycode.profiling import profile
from mlprodict.sklapi import OnnxSpeedupTransformer
from cpyquickhelper.numbers.speed_measure import measure_time
from tqdm import tqdm

################################
# Data and models to test.

data, _ = make_regression(1000, n_features=20)
data = data.astype(numpy.float32)
models = [
    ('sklearn', PCA(n_components=10)),
    ('python', OnnxSpeedupTransformer(
        PCA(n_components=10), runtime='python')),
    ('onnxruntime1', OnnxSpeedupTransformer(
        PCA(n_components=10), runtime='onnxruntime1')),
    ('numpy', OnnxSpeedupTransformer(
        PCA(n_components=10), runtime='numpy')),
    ('numba', OnnxSpeedupTransformer(
        PCA(n_components=10), runtime='numba'))]

#################################
# Training.

for name, model in tqdm(models):
    model.fit(data)

#################################
# Profiling of runtime `onnxruntime1`.


def fct():
    for i in range(1000):
        models[2][1].transform(data)


res = profile(fct, pyinst_format="text")
print(res[1])


#################################
# Profiling of runtime `numpy`.

def fct():
    for i in range(1000):
        models[3][1].transform(data)


res = profile(fct, pyinst_format="text")
print(res[1])

#################################
# The class *OnnxSpeedupTransformer* converts the PCA
# into ONNX and then converts it into a python code using
# *numpy*. The code is the following.

print(models[3][1].numpy_code_)

#################################
# Benchmark.

bench = []
for name, model in tqdm(models):
    for size in (1, 10, 100, 1000, 10000, 100000, 200000):
        data, _ = make_regression(size, n_features=20)
        data = data.astype(numpy.float32)

        # We run it a first time (numba compiles
        # the function during the first execution).
        model.transform(data)
        res = measure_time(
            lambda: model.transform(data), div_by_number=True,
            context={'data': data, 'model': model})
        res['name'] = name
        res['size'] = size
        bench.append(res)

df = DataFrame(bench)
piv = df.pivot("size", "name", "average")
piv

######################################
# Graph.
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
piv.plot(title="Speedup PCA with ONNX (lower better)",
         logx=True, logy=True, ax=ax[0])
piv2 = piv.copy()
for c in piv2.columns:
    piv2[c] /= piv['sklearn']
print(piv2)
piv2.plot(title="baseline=scikit-learn (lower better)",
          logx=True, logy=True, ax=ax[1])
plt.show()
