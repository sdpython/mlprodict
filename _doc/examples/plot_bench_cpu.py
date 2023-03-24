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
import numpy
import matplotlib.pyplot as plt
from pyquickhelper.loghelper import run_cmd
from pandas import DataFrame, concat
from mlprodict.testing.experimental_c_impl.experimental_c import (
    benchmark_cache, benchmark_cache_tree)

obs = []
step = 2**12
for i in tqdm(range(step, 2**20 + step, step)):
    res = min([benchmark_cache(i, False), benchmark_cache(
        i, False), benchmark_cache(i, False)])
    if res < 0:
        # overflow
        continue
    obs.append(dict(size=i, perf=res))

df = DataFrame(obs)
mean = df.perf.mean()
lag = 32
for i in range(2, df.shape[0]):
    df.loc[i, "smooth"] = df.loc[i - 8:i + 8, "perf"].median()
    if i > lag and i < df.shape[0] - lag:
        df.loc[i, "delta"] = (
            mean +
            df.loc[i:i + lag, "perf"].mean() -
            df.loc[i - lag + 1:i + 1, "perf"]).mean()

###########################################
# Cache size estimator
# ++++++++++++++++++++

cache_size_index = int(df.delta.argmax())
cache_size = df.loc[cache_size_index, "size"] * 2
print(f"L2 cache size estimation is {cache_size / 2 ** 20:1.3f} Mb.")

###########################################
# Verification
# ++++++++++++

try:
    out, err = run_cmd("lscpu", wait=True)
    print("\n".join(_ for _ in out.split("\n") if "cache:" in _))
except Exception as e:
    print(f"failed due to {e}")

df = df.set_index("size")
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
df.plot(ax=ax, title="Cache Performance time/size", logy=True)
fig.savefig("plot_benchmark_cpu_array.png")

#####################################
# TreeEnsemble Performance
# ++++++++++++++++++++++++
#
# We simulate the computation of a TreeEnsemble
# of 50 features, 100 trees and depth of 10
# (so :math:`2^10` nodes.)

dfs = []
cols = []
drop = []
for n in tqdm(range(10)):
    res = benchmark_cache_tree(
        n_rows=2000, n_features=50, n_trees=100,
        tree_size=1024, max_depth=10, search_step=64)
    res = [[max(r.row, i), r.time] for i, r in enumerate(res)]
    df = DataFrame(res)
    df.columns = [f"i{n}", f"time{n}"]
    dfs.append(df)
    cols.append(df.columns[-1])
    drop.append(df.columns[0])

df = concat(dfs, axis=1).reset_index(drop=True)
df["i"] = df["i0"]
df = df.drop(drop, axis=1)
df["time_avg"] = df[cols].mean(axis=1)
df["time_med"] = df[cols].median(axis=1)

df.head()

##########################################
# Estimation
# ++++++++++

print(f"Optimal batch size is among:")
dfi = df[["time_med", "i"]].groupby("time_med").min()
dfi_min = set(dfi["i"])
dfsub = df[df["i"].isin(dfi_min)]
dfs = dfsub.sort_values("time_med").reset_index()
print(dfs[["i", "time_med", "time_avg"]].head(10))

################################################
# One possible estimation

subdfs = dfs[:20]
avg = (subdfs["i"] / subdfs["time_avg"]).sum() / (subdfs["time_avg"] ** (-1)).sum()
print(f"Estimation: {avg}")

##############################################
# Plots.

cols_time = ["time_avg", "time_med"]
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
df.set_index("i").drop(cols_time, axis=1).plot(
    ax=ax[0], title="TreeEnsemble Performance time per row",
    logy=True, linewidth=0.2)
df.set_index("i")[cols_time].plot(ax=ax[1], linewidth=1., logy=True)
fig.savefig("plot_benchmark_cpu_tree.png")
