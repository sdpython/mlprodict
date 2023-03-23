"""
.. _l-example-bench-cpu:

Measuring CPU performance
=========================

Processor caches must be taken into account when writing an algorithm,
see `Memory part 2: CPU caches <https://lwn.net/Articles/252125/>`_
from Ulrich Drepper.

Function 1
++++++++++
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
from mlprodict.testing.experimental_c_impl.experimental_c import benchmark_cache

obs = []
step = 2**12
for i in tqdm(range(step, 2**24 + step, step)):
    res = benchmark_cache(i, False)
    if res < 0:
        # overflow
        continue
    obs.append(dict(size=i, perf=res))

df = DataFrame(obs).set_index("size")

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
df.plot(ax=ax, title="Cache Performance time/size", logy=True)
fig.savefig("plot_benchmark_cpu.png")
