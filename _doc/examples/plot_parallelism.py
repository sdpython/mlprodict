"""
.. _l-example-parallelism:

When to parallelize?
====================

That is the question. Parallize computation
takes some time to set up, it is not the right
solution in every case. The following example studies
the parallelism introduced into the runtime of
*TreeEnsembleRegressor* to see when it is best
to do it.

.. contents::
    :local:

"""
from pprint import pprint
import numpy
from pandas import DataFrame
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import config_context
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from cpyquickhelper.numbers import measure_time
from pyquickhelper.pycode.profiling import profile
from mlprodict.onnx_conv import to_onnx, register_rewritten_operators
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.model_info import analyze_model

#####################################
# Available optimisations on this machine.

from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
print(code_optimisation())


########################################
# Training and converting a model
# +++++++++++++++++++++++++++++++


data = make_regression(50000, 20)
X, y = data
X_train, X_test, y_train, y_test = train_test_split(X, y)

hgb = HistGradientBoostingRegressor(max_iter=100, max_depth=6)
hgb.fit(X_train, y_train)
print(hgb)

########################################
# Let's get more statistics about the model itself.
pprint(analyze_model(hgb))

#################################
# And let's convert it.

register_rewritten_operators()
onx = to_onnx(hgb, X_train[:1].astype(numpy.float32))
oinf = OnnxInference(onx, runtime='python_compiled')
print(oinf)


################################
# The runtime of the forest is in the following object.

print(oinf.sequence_[0].ops_)
print(oinf.sequence_[0].ops_.rt_)

########################################
# And the threshold used to start parallelizing
# based on the number of observations.

print(oinf.sequence_[0].ops_.rt_.omp_N_)


######################################
# Profiling
# +++++++++
#
# This step involves :epkg:`pyinstrument` to measure
# where the time is spent. Both :epkg:`scikit-learn`
# and :epkg:`mlprodict` runtime are called so that
# the prediction times can be compared.

X32 = X_test.astype(numpy.float32)


def runlocal():
    with config_context(assume_finite=True):
        for i in range(0, 100):
            oinf.run({'X': X32[:1000]})
            hgb.predict(X_test[:1000])


print("profiling...")
txt = profile(runlocal, pyinst_format='text')
print(txt[1])

##########################################
# Now let's measure the performance the average
# computation time per observations for 2 to 100
# observations. The runtime implemented in
# :epkg:`mlprodict` parallizes the computation
# after a given number of observations.

obs = []
for N in tqdm(list(range(2, 21))):
    m = measure_time("oinf.run({'X': x})",
                     {'oinf': oinf, 'x': X32[:N]},
                     div_by_number=True,
                     number=20)
    m['N'] = N
    m['RT'] = 'ONNX'
    obs.append(m)

    with config_context(assume_finite=True):
        m = measure_time("hgb.predict(x)",
                         {'hgb': hgb, 'x': X32[:N]},
                         div_by_number=True,
                         number=15)
    m['N'] = N
    m['RT'] = 'SKL'
    obs.append(m)

df = DataFrame(obs)
num = ['min_exec', 'average', 'max_exec']
for c in num:
    df[c] /= df['N']
df.head()

#######################################
# Graph.

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df[df.RT == 'ONNX'].set_index('N')[num].plot(ax=ax[0])
ax[0].set_title("Average ONNX prediction time per observation in a batch.")
df[df.RT == 'SKL'].set_index('N')[num].plot(ax=ax[1])
ax[1].set_title(
    "Average scikit-learn prediction time\nper observation in a batch.")


######################################
# Gain from parallelization
# +++++++++++++++++++++++++
#
# There is a clear gap between after and before 10 observations
# when it is parallelized. Does this threshold depends on the number
# of trees in the model?
# For that we compute for each model the average prediction time
# up to 10 and from 10 to 20.

def parallized_gain(df):
    df = df[df.RT == 'ONNX']
    df10 = df[df.N <= 10]
    t10 = sum(df10['average']) / df10.shape[0]
    df10p = df[df.N > 10]
    t10p = sum(df10p['average']) / df10p.shape[0]
    return t10 / t10p


print('gain', parallized_gain(df))

#################################
# Measures based on the number of trees
# +++++++++++++++++++++++++++++++++++++
#
# We trained many models with different number
# of trees to see how the parallelization gain
# is moving. One models is trained for every
# distinct number of trees and then the prediction
# time is measured for different number of observations.

tries_set = [2, 5, 8] + list(range(10, 50, 5)) + list(range(50, 101, 10))
tries = [(nb, N) for N in range(2, 21, 2) for nb in tries_set]

###########################
# training

models = {100: (hgb, oinf)}
for nb in tqdm(set(_[0] for _ in tries)):
    if nb not in models:
        hgb = HistGradientBoostingRegressor(max_iter=nb, max_depth=6)
        hgb.fit(X_train, y_train)
        onx = to_onnx(hgb, X_train[:1].astype(numpy.float32))
        oinf = OnnxInference(onx, runtime='python_compiled')
        models[nb] = (hgb, oinf)

###########################
# prediction time

obs = []

for nb, N in tqdm(tries):
    hgb, oinf = models[nb]
    m = measure_time("oinf.run({'X': x})",
                     {'oinf': oinf, 'x': X32[:N]},
                     div_by_number=True,
                     number=50)
    m['N'] = N
    m['nb'] = nb
    m['RT'] = 'ONNX'
    obs.append(m)

df = DataFrame(obs)
num = ['min_exec', 'average', 'max_exec']
for c in num:
    df[c] /= df['N']
df.head()

##########################
# Let's compute the gains.

gains = []
for nb in set(df['nb']):
    gain = parallized_gain(df[df.nb == nb])
    gains.append(dict(nb=nb, gain=gain))

dfg = DataFrame(gains)
dfg = dfg.sort_values('nb').reset_index(drop=True).copy()
dfg

##########################################
# Graph.

ax = dfg.set_index('nb').plot()
ax.set_title(
    "Parallelization gain depending\non the number of trees\n(max_depth=6).")

##############################
# That does not answer the question we are looking for
# as we would like to know the best threshold *th*
# which defines the number of observations for which
# we should parallelized. This number depends on the number
# of trees. A gain > 1 means the parallization should happen
# Here, even two observations is ok.
# Let's check with lighter trees (``max_depth=2``),
# maybe in that case, the conclusion is different.

models = {100: (hgb, oinf)}
for nb in tqdm(set(_[0] for _ in tries)):
    if nb not in models:
        hgb = HistGradientBoostingRegressor(max_iter=nb, max_depth=2)
        hgb.fit(X_train, y_train)
        onx = to_onnx(hgb, X_train[:1].astype(numpy.float32))
        oinf = OnnxInference(onx, runtime='python_compiled')
        models[nb] = (hgb, oinf)

obs = []
for nb, N in tqdm(tries):
    hgb, oinf = models[nb]
    m = measure_time("oinf.run({'X': x})",
                     {'oinf': oinf, 'x': X32[:N]},
                     div_by_number=True,
                     number=50)
    m['N'] = N
    m['nb'] = nb
    m['RT'] = 'ONNX'
    obs.append(m)

df = DataFrame(obs)
num = ['min_exec', 'average', 'max_exec']
for c in num:
    df[c] /= df['N']
df.head()

###################################
# Measures.

gains = []
for nb in set(df['nb']):
    gain = parallized_gain(df[df.nb == nb])
    gains.append(dict(nb=nb, gain=gain))

dfg = DataFrame(gains)
dfg = dfg.sort_values('nb').reset_index(drop=True).copy()
dfg

#######################################
# Graph.

ax = dfg.set_index('nb').plot()
ax.set_title(
    "Parallelization gain depending\non the number of trees\n(max_depth=3).")

#########################################
# The conclusion is somewhat the same but
# it shows that the bigger the number of trees is
# the bigger the gain is and under the number of
# cores of the processor.
#
# Moving the theshold
# +++++++++++++++++++
#
# The last experiment consists in comparing the prediction
# time with or without parallelization for different
# number of observation.

hgb = HistGradientBoostingRegressor(max_iter=40, max_depth=6)
hgb.fit(X_train, y_train)
onx = to_onnx(hgb, X_train[:1].astype(numpy.float32))
oinf = OnnxInference(onx, runtime='python_compiled')


obs = []
for N in tqdm(list(range(2, 51))):
    oinf.sequence_[0].ops_.rt_.omp_N_ = 100
    m = measure_time("oinf.run({'X': x})",
                     {'oinf': oinf, 'x': X32[:N]},
                     div_by_number=True,
                     number=20)
    m['N'] = N
    m['RT'] = 'ONNX'
    m['PARALLEL'] = False
    obs.append(m)

    oinf.sequence_[0].ops_.rt_.omp_N_ = 1
    m = measure_time("oinf.run({'X': x})",
                     {'oinf': oinf, 'x': X32[:N]},
                     div_by_number=True,
                     number=50)
    m['N'] = N
    m['RT'] = 'ONNX'
    m['PARALLEL'] = True
    obs.append(m)

df = DataFrame(obs)
num = ['min_exec', 'average', 'max_exec']
for c in num:
    df[c] /= df['N']
df.head()

##########################################
# Graph.

piv = df[['N', 'PARALLEL', 'average']].pivot('N', 'PARALLEL', 'average')
ax = piv.plot(logy=True)
ax.set_title("Prediction time with and without parallelization.")

###############################
# Parallelization is working.


plt.show()
