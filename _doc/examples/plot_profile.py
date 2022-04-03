"""
.. _l-example-profile:

Profile the execution of a runtime
==================================

The following example shows how to profile the execution
of a model with different runtime.

.. contents::
    :local:

Training and converting a model
+++++++++++++++++++++++++++++++
"""

import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from pyquickhelper.pycode.profiling import profile
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict import get_ir_version

data = load_boston()
X, y = data.data, data.target

dt = DecisionTreeRegressor()
dt.fit(X, y)
onx = to_onnx(dt, X[:1].astype(numpy.float32), target_opset=11)
oinf = OnnxInference(onx, runtime='python_compiled')
print(oinf)


###################################
# Profiling and comparison with scikit-learn
# ++++++++++++++++++++++++++++++++++++++++++

X32 = X.astype(numpy.float32)


def runlocaldt():
    for i in range(0, 5000):
        oinf.run({'X': X32[:10]})
        dt.predict(X[:10])


print("profiling...")
txt = profile(runlocaldt, pyinst_format='text')
print(txt[1])


###########################################
# Profiling for AdaBoostRegressor
# +++++++++++++++++++++++++++++++
#
# The next example shows how long the python runtime
# spends in each operator.

ada = AdaBoostRegressor()
ada.fit(X, y)
onx = to_onnx(ada, X[:1].astype(numpy.float32), target_opset=11)
oinf = OnnxInference(onx, runtime='python_compiled')
print(oinf)

########################################
# The profiling.


def runlocal():
    for i in range(0, 500):
        oinf.run({'X': X32})


print("profiling...")
txt = profile(runlocal, pyinst_format='text')
print(txt[1])

###########################################
# With a different runtime
# ++++++++++++++++++++++++
#
# Let's compare to :epkg:`onnxruntime`.

onx.ir_version = get_ir_version(11)
oinf = OnnxInference(onx, runtime='onnxruntime1')


def runlocalort():
    for i in range(0, 500):
        oinf.run({'X': X32})


print("profiling with onnxruntime...")
txt = profile(runlocalort, pyinst_format='text')
print(txt[1])

################################
# py-spy
# ++++++
#
# :epkg:`py-spy` may be used to dig into native
# functions. An example can be found at:
# `Profiling AdaBoostRegressor
# <http://www.xavierdupre.fr/app/_benchmarks/
# helpsphinx/onnx/onnx_profiling_reg_adaboost.html#
# profiling-adaboostregressor>`_.
# The last piece of code uses the standard
# python profiler.

pr, df = profile(runlocal, as_df=True)

ax = df[['namefct', 'cum_tall']].head(n=15).set_index(
    'namefct').plot(kind='bar', figsize=(8, 3), rot=15)
ax.set_title("Simple profiling")
for la in ax.get_xticklabels():
    la.set_horizontalalignment('right')
plt.show()
