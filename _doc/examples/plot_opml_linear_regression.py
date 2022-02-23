"""
.. _l-example-linear-regression-bench:

Benchmark Linear Regression
===========================

The script compares different implementations for the operator
LinearRegression.

* *baseline*: LinearRegression from :epkg:`scikit-learn`
* *ort*: :epkg:`onnxruntime`,
* *mlprodict*: an implementation based on an array of structures,
  every structure describes a node,

.. contents::
    :local:

Import
++++++
"""
import warnings
from time import perf_counter as time
from multiprocessing import cpu_count
import numpy
from numpy.random import rand
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
import pandas
from onnxruntime import InferenceSession
from sklearn import config_context
from sklearn.linear_model import LinearRegression
from sklearn.utils._testing import ignore_warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt import OnnxInference

############################################
# Available optimisation on this machine.

from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
print(code_optimisation())


###################################
# Versions
# ++++++++


def version():
    from datetime import datetime
    import sklearn
    import numpy
    import onnx
    import onnxruntime
    import skl2onnx
    import mlprodict
    df = pandas.DataFrame([
        {"name": "date", "version": str(datetime.now())},
        {"name": "numpy", "version": numpy.__version__},
        {"name": "scikit-learn", "version": sklearn.__version__},
        {"name": "onnx", "version": onnx.__version__},
        {"name": "onnxruntime", "version": onnxruntime.__version__},
        {"name": "skl2onnx", "version": skl2onnx.__version__},
        {"name": "mlprodict", "version": mlprodict.__version__},
    ])
    return df


version()


##############################
# Implementations to benchmark
# ++++++++++++++++++++++++++++

def fcts_model(X, y, n_jobs):
    "LinearRegression."
    model = LinearRegression(n_jobs=n_jobs)
    model.fit(X, y)

    initial_types = [('X', FloatTensorType([None, X.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_types)
    sess = InferenceSession(onx.SerializeToString())
    outputs = [o.name for o in sess.get_outputs()]
    oinf = OnnxInference(onx, runtime="python")

    def predict_skl_predict(X, model=model):
        return model.predict(X)

    def predict_onnxrt_predict(X, sess=sess):
        return sess.run(outputs[:1], {'X': X})[0]

    def predict_onnx_inference(X, oinf=oinf):
        return oinf.run({'X': X})["variable"]

    return {'predict': (
        predict_skl_predict, predict_onnxrt_predict,
        predict_onnx_inference)}


##############################
# Benchmarks
# ++++++++++

def allow_configuration(**kwargs):
    return True


def bench(n_obs, n_features, n_jobss,
          methods, repeat=10, verbose=False):
    res = []
    for nfeat in n_features:

        ntrain = 50000
        X_train = numpy.empty((ntrain, nfeat)).astype(numpy.float32)
        X_train[:, :] = rand(ntrain, nfeat)[:, :]
        eps = rand(ntrain) - 0.5
        y_train = X_train.sum(axis=1) + eps

        for n_jobs in n_jobss:
            fcts = fcts_model(X_train, y_train, n_jobs)

            for n in n_obs:
                for method in methods:

                    fct1, fct2, fct3 = fcts[method]

                    if not allow_configuration(n=n, nfeat=nfeat,
                                               n_jobs=n_jobs, method=method):
                        continue

                    obs = dict(n_obs=n, nfeat=nfeat, method=method,
                               n_jobs=n_jobs)

                    # creates different inputs to avoid caching in any ways
                    Xs = []
                    for r in range(repeat):
                        x = numpy.empty((n, nfeat))
                        x[:, :] = rand(n, nfeat)[:, :]
                        Xs.append(x.astype(numpy.float32))

                    # measures the baseline
                    with config_context(assume_finite=True):
                        st = time()
                        repeated = 0
                        for X in Xs:
                            p1 = fct1(X)
                            repeated += 1
                            if time() - st >= 1:
                                break  # stops if longer than a second
                        end = time()
                        obs["time_skl"] = (end - st) / repeated

                    # measures the new implementation
                    st = time()
                    r2 = 0
                    for X in Xs:
                        p2 = fct2(X)
                        r2 += 1
                        if r2 >= repeated:
                            break
                    end = time()
                    obs["time_ort"] = (end - st) / r2

                    # measures the other new implementation
                    st = time()
                    r2 = 0
                    for X in Xs:
                        p2 = fct3(X)
                        r2 += 1
                        if r2 >= repeated:
                            break
                    end = time()
                    obs["time_mlprodict"] = (end - st) / r2

                    # final
                    res.append(obs)
                    if verbose and (len(res) % 1 == 0 or n >= 10000):
                        print("bench", len(res), ":", obs)

                    # checks that both produce the same outputs
                    if n <= 10000:
                        if len(p1.shape) == 1 and len(p2.shape) == 2:
                            p2 = p2.ravel()
                        try:
                            assert_almost_equal(
                                p1.ravel(), p2.ravel(), decimal=5)
                        except AssertionError as e:
                            warnings.warn(str(e))
    return res

#########################################
# Graphs
# ++++++


def plot_rf_models(dfr):

    def autolabel(ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('%1.1fx' % height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)

    engines = [_.split('_')[-1] for _ in dfr.columns if _.startswith("time_")]
    engines = [_ for _ in engines if _ != 'skl']
    for engine in engines:
        dfr["speedup_%s" % engine] = dfr["time_skl"] / dfr["time_%s" % engine]
    print(dfr.tail().T)

    ncols = 2
    fig, axs = plt.subplots(len(engines), ncols, figsize=(
        14, 4 * len(engines)), sharey=True)

    row = 0
    for row, engine in enumerate(engines):
        pos = 0
        name = "LinearRegression - %s" % engine
        for nf in sorted(set(dfr.nfeat)):
            for n_jobs in sorted(set(dfr.n_jobs)):
                sub = dfr[(dfr.nfeat == nf) & (dfr.n_jobs == n_jobs)]
                ax = axs[row, pos]
                labels = sub.n_obs
                means = sub["speedup_%s" % engine]

                x = numpy.arange(len(labels))
                width = 0.90

                rects1 = ax.bar(x, means, width, label='Speedup')
                if pos == 0:
                    ax.set_yscale('log')
                    ax.set_ylim([0.1, max(dfr["speedup_%s" % engine])])

                if pos == 0:
                    ax.set_ylabel('Speedup')
                ax.set_title('%s\n%d features\n%d jobs' % (name, nf, n_jobs))
                if row == len(engines) - 1:
                    ax.set_xlabel('batch size')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                autolabel(ax, rects1)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(8)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(8)
                pos += 1

    fig.tight_layout()
    return fig, ax


###################################
# Run benchs
# ++++++++++

@ignore_warnings(category=FutureWarning)
def run_bench(repeat=250, verbose=False):
    n_obs = [1, 10, 100, 1000, 10000]
    methods = ['predict']
    n_features = [10, 50]
    n_jobss = [cpu_count()]

    start = time()
    results = bench(n_obs, n_features, n_jobss,
                    methods, repeat=repeat, verbose=verbose)
    end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec cpu=%d\n" % (end - start, cpu_count()))

    # plot the results
    return results_df


name = "plot_linear_regression"
df = run_bench(verbose=True)
df.to_csv("%s.csv" % name, index=False)
df.to_excel("%s.xlsx" % name, index=False)
fig, ax = plot_rf_models(df)
fig.savefig("%s.png" % name)
plt.show()
