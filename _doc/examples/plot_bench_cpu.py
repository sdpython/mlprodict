"""
.. _l-example-bench-cpu:

Measuring CPU performance
=========================

Processor caches must be taken into account when writing an algorithm,
see `Memory part 2: CPU caches <https://lwn.net/Articles/252125/>`_
from Ulrich Drepper.

Cache Performance
+++++++++++++++++
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
from mlprodict.testing.experimental_c_impl.experimental_c import (
    benchmark_cache, benchmark_cache_tree)

obs = []
step = 2**12
for i in tqdm(range(step, 2**21 + step, step)):
    res = benchmark_cache(i, False)
    if res < 0:
        # overflow
        continue
    obs.append(dict(size=i, perf=res))

df = DataFrame(obs).set_index("size")

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
df.plot(ax=ax, title="Cache Performance time/size", logy=True)
fig.savefig("plot_benchmark_cpu_array.png")

#####################################
# TreeEnsemble Performance
# ++++++++++++++++++++++++

res = benchmark_cache_tree(1000)
res = [[r.n_trial, r.row, r.time] for r in res]
print(res[:10])
print(res[-10:])

df = DataFrame(res)
df.columns = ["n", "i", "time"]
df = df.set_index("i")
print(df.tail())

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
df.plot(ax=ax, title="TreeEnsemble Performance time per row", logy=True)
fig.savefig("plot_benchmark_cpu_tree.png")



