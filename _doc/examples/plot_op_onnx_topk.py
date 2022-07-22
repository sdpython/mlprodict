"""
.. _onnxtopkrst:

TopK benchmark
==============

This example compares :epkg:`onnxruntime` and :epkg:`mlprodict`
for an implementation of operator `TopK
<https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK>`_.
We measure two runtimes by computing a ratio between their
time execution through the following kind of graphs.

.. contents::
    :local:

Graph to compare performance
++++++++++++++++++++++++++++
"""

from numpy.random import randn
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame
from onnxruntime import InferenceSession, __version__ as ort_version
from tqdm import tqdm
from cpyquickhelper.numbers import measure_time
from pyquickhelper.pycode.profiling import profile
from skl2onnx.algebra.onnx_ops import OnnxTopK_11
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxTopK
from mlprodict.onnxrt.validate.validate_benchmark import benchmark_fct
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.ops_cpu.op_topk import (
    topk_sorted_implementation, topk_sorted_implementation_cpp)
from mlprodict import __version__ as mlp_version
from mlprodict.plotting.plotting import plot_benchmark_metrics

############################################
# Available optimisation on this machine.

from mlprodict.testing.experimental_c_impl.experimental_c import code_optimisation
print(code_optimisation())

###########################################
# Graph.


def plot_metric(metric, ax=None, xlabel="N", ylabel="k", middle=1.,
                transpose=False, shrink=1.0, title=None):
    ax, cbar = plot_benchmark_metrics(
        metric, ax=ax, xlabel=xlabel, ylabel=ylabel, middle=middle,
        transpose=transpose, cbar_kw={'shrink': shrink})
    if title is not None:
        ax.set_title(title)
    return ax


data = {(1, 1): 0.1, (10, 1): 1, (1, 10): 2,
        (10, 10): 100, (100, 1): 100, (100, 10): 1000}

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_metric(data, ax[0], shrink=0.6)

##################################
#

plot_metric(data, ax[1], transpose=True)


##############################
# TopK in ONNX
# ++++++++++++
#
# The following lines creates an ONNX graph using
# one TopK ONNX node. The outcome is the ONNX graph
# converted into json.


X32 = randn(100000, 100).astype(numpy.float32)

node = OnnxTopK_11('X', numpy.array([5], dtype=numpy.int64),
                   output_names=['dist', 'ind'])

model_onnx = node.to_onnx(
    [('X', X32)], target_opset=12,
    # shape inference does not seem to work in onnxruntime
    # so we speccify the output shape
    outputs=[('dist', X32[:1, :5]),
             ('ind', X32[:1, :5].astype(numpy.int64))])
model_onnx


####################################
# That gives...


oinf = OnnxInference(model_onnx, runtime="python")
res = oinf.run({'X': X32})
dist, ind = res['dist'], res['ind']
dist[:2], ind[:2]

##########################################
# With onnxruntime.


sess = InferenceSession(model_onnx.SerializeToString())
dist, ind = sess.run(None, {'X': X32})
dist[:2], ind[:2]


########################################
# Let's compare two implementations.


def benchmark(X, fct1, fct2, N, repeat=10, number=10):

    def ti(n):
        if n <= 1:
            return 50
        if n <= 1000:
            return 2
        if n <= 10000:
            return 0.51
        return 0.11

    # to warm up the engine
    time_kwargs = {n: dict(repeat=10, number=10) for n in N[:2]}
    benchmark_fct(fct1, X, time_kwargs=time_kwargs, skip_long_test=False)
    benchmark_fct(fct2, X, time_kwargs=time_kwargs, skip_long_test=False)
    # real measure
    time_kwargs = {n: dict(repeat=int(repeat * ti(n)),
                           number=int(number * ti(n))) for n in N}
    res1 = benchmark_fct(fct1, X, time_kwargs=time_kwargs,
                         skip_long_test=False)
    res2 = benchmark_fct(fct2, X, time_kwargs=time_kwargs,
                         skip_long_test=False)

    res = {}
    for r in sorted(res1):
        r1 = res1[r]
        r2 = res2[r]
        ratio = r2['ttime'] / r1['ttime']
        res[r] = ratio
    return res


N = [1, 10, 100, 1000, 10000, 100000]
res = benchmark(X32, lambda x: sess.run(None, {'X': x}),
                lambda x: oinf.run({'X': x}), N=N)
res


#########################################
# The implementation in `mlprodict
# <https://github.com/sdpython/mlprodict/blob/master/
# mlprodict/onnxrt/ops_cpu/_op_onnx_numpy.cpp#L246>`_
# is faster when the number of rows grows. It is faster
# for 1 rows, for many rows, the implementation
# uses openmp to parallelize.
#
# C++ implementation vs numpy
# +++++++++++++++++++++++++++
#
# :epkg:`scikit-learn` uses :epkg:`numpy` to compute the top *k* elements.


res = benchmark(X32, lambda x: topk_sorted_implementation(x, 5, 1, 0),
                lambda x: topk_sorted_implementation_cpp(x, 5, 1, 0), N=N)
res


###########################################
# It seems to be faster too. Let's profile.


xr = randn(1000000, 100)
text = profile(lambda: topk_sorted_implementation(xr, 5, 1, 0),
               pyinst_format='text')[1]
print(text)

####################################
# Parallelisation
# +++++++++++++++
#
# We need to disable the parallelisation to
# really compare both implementation.

# In[11]:


def benchmark_test(X, fct1, fct2, N, K, repeat=10, number=10):
    res = {}
    for k in tqdm(K):
        def f1(x, k=k): return fct1(x, k=k)
        def f2(x, k=k): return fct2(x, k=k)
        r = benchmark(X32, f1, f2, N=N, repeat=repeat, number=number)
        for n, v in r.items():
            res[n, k] = v
    return res


K = [1, 2, 5, 10, 15]
N = [1, 2, 3, 10, 100, 1000, 10000]

bench_para = benchmark_test(
    X32, (lambda x, k: topk_sorted_implementation_cpp(
        x, k=k, axis=1, largest=0, th_para=100000000)),
    (lambda x, k: topk_sorted_implementation_cpp(
        x, k=k, axis=1, largest=0, th_para=1)),
    N=N, K=K)

bench_para


#######################################
# As a graph.


plot_metric(bench_para, transpose=False, title="TopK and parallelisation\n"
            "< 1 means parallelisation is faster", shrink=0.75)

###############################################
# This is somehow expected. First column is closed to
# 1 as it is the same code being compared. Next columns
# are red, meaning the parallelisation does not help,
# the parallelisation helps for bigger N, as least more than 100.
#
# Parallellisation with ONNX
# ++++++++++++++++++++++++++
#
# We replicate the same experiment with an ONNX graph.


k_ = numpy.array([3], dtype=numpy.int64)
node = OnnxTopK_11('X', 'k',
                   output_names=['dist', 'ind'])

model_onnx = node.to_onnx(
    [('X', X32), ('k', k_)], target_opset=12,
    # shape inference does not seem to work in onnxruntime
    # so we speccify the output shape
    outputs=[('dist', X32[:1, :5]),
             ('ind', X32[:1, :5].astype(numpy.int64))])


#################################
# Test


oinf_no_para = OnnxInference(model_onnx, runtime="python")
res = oinf_no_para.run({'X': X32, 'k': k_})
dist, ind = res['dist'], res['ind']
dist[:2], ind[:2]


########################################
# Let's play with the thresholds triggering the parallelisation.

oinf_para = OnnxInference(model_onnx, runtime="python")
oinf_no_para.sequence_[0].ops_.th_para = 100000000
oinf_para.sequence_[0].ops_.th_para = 1


##################################
# Results.


bench_onnx_para = benchmark_test(
    X32, (lambda x, k: oinf_no_para.run(
        {'X': x, 'k': numpy.array([k], dtype=numpy.int64)})),
    (lambda x, k: oinf_para.run(
        {'X': x, 'k': numpy.array([k], dtype=numpy.int64)})),
    N=N, K=K)
bench_onnx_para


#################################
# As a graph.


plot_metric(bench_onnx_para, transpose=False,
            title="TopK and parallelisation with ONNX\n< 1 means "
            "parallelisation is faster", shrink=0.75)

#########################################
# Pretty much the same results.
#
# onnxruntime vs mlprodict (no parallelisation)
# +++++++++++++++++++++++++++++++++++++++++++++

sess = InferenceSession(model_onnx.SerializeToString())


bench_ort = benchmark_test(
    X32, (lambda x, k: sess.run(
        None, {'X': x, 'k': numpy.array([k], dtype=numpy.int64)})),
    (lambda x, k: oinf_no_para.run(
        {'X': x, 'k': numpy.array([k], dtype=numpy.int64)})),
    N=N, K=K)
bench_ort

######################################
# As a graph.

plot_metric(bench_ort, transpose=False,
            title="TopK, onnxruntime vs mlprodict\n< 1 means mlprodict "
            "is faster\nno parallelisation", shrink=0.75)

######################################
# It seems the implementation of operator TopK in
# onnxruntime 1.1.1 can be improved.
#
# Versions:
ort_version, mlp_version

#########################################
# And with parallelisation above 50 rows.

oinf_para.sequence_[0].ops_.th_para = 50
bench_ort_para = benchmark_test(
    X32, (lambda x, k: sess.run(
        None, {'X': x, 'k': numpy.array([k], dtype=numpy.int64)})),
    (lambda x, k: oinf_para.run(
        {'X': x, 'k': numpy.array([k], dtype=numpy.int64)})),
    N=N, K=K)
bench_ort_para


###########################################
# As a graph.


plot_metric(bench_ort_para, transpose=False,
            title="TopK, onnxruntime vs mlprodict\n< 1 means mlprodict "
            "is faster\nparallelisation above 50 rows", shrink=0.75)

#################################
# onnxruntime and mlprodict implement the same algorithm.
# The only difference comes from the threshold which
# trigger the parallelisation. It should depend on the machine.
# That explains the difference in time for 100 observations.
#
##############################
# Interesting...
#
# Comparison with onnxruntime
# +++++++++++++++++++++++++++


X = numpy.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
], dtype=numpy.float32)

K = numpy.array([3], dtype=numpy.int64)


node = OnnxTopK('X', K, output_names=['values', 'indices'],
                op_version=12)
onx = node.to_onnx([('X', FloatTensorType())])

py_topk = OnnxInference(onx, runtime="python_compiled")
ort_topk = InferenceSession(onx.SerializeToString())


##################################
# Check the outputs.


r1 = py_topk.run({'X': X})
r1


###########################
#

r2 = ort_topk.run(None, {'X': X})
r2


#################################
# Some figures.

bs = []
bs.append(measure_time(lambda: py_topk.run({'X': X}),
                       context=globals(), div_by_number=True))
bs[-1]['c'] = 'py'
bs[-1]

#################################
#

bs.append(measure_time(lambda: ort_topk.run(None, {'X': X}),
                       context=globals(), div_by_number=True))
bs[-1]['c'] = 'or'
bs[-1]

#####################################
#

X = numpy.random.randn(10000, 100).astype(numpy.float32)


bs.append(measure_time(lambda: py_topk.run({'X': X}),
                       context=globals(), div_by_number=True))
bs[-1]['c'] = 'py-100'
bs[-1]


#####################################
#


bs.append(measure_time(lambda: ort_topk.run(None, {'X': X}),
                       context=globals(), div_by_number=True))
bs[-1]['c'] = 'ort-100'
bs[-1]

#####################################
#

df = DataFrame(bs)
df
