"""
.. _l-op-merge:

Merges benchmarks
=================

This script merges benchmark from :ref:`l-b-reducesummax`,
:ref:`l-b-reducesummean`, :ref:`l-b-reducesum`.

.. contents::
    :local:

Reads data
++++++++++

One file looks like this:
"""
import os
import pandas
import matplotlib.pyplot as plt

df = pandas.read_excel("keep_plot_reducesum_master.xlsx")
df.head(n=4).T

#############################################
# The other files.

index = ['fct', 'axes', 'N', 'shape']
value = ['average']
files = [
    ('ReduceSum', 'keep_plot_reducesum_master.xlsx', 'keep_plot_reducesum.xlsx'),
    ('ReduceMax', 'plot_reducemax_master.xlsx', 'plot_reducemax.xlsx'),
    ('ReduceMean', 'plot_reducemean_master.xlsx', 'plot_reducemean.xlsx'),
]

merged = []
for title, ref, impl in files:
    if not os.path.exists(ref) or not os.path.exists(impl):
        continue
    df1 = pandas.read_excel(ref)
    df2 = pandas.read_excel(impl)
    df1 = df1[index + value]
    df2 = df2[index + value]
    merge = df1.merge(df2, on=index, suffixes=('_ref', '_new'))
    merge['op'] = title
    merge['SpeedUp'] = merge[value[0] + "_ref"] / merge[value[0] + "_new"]
    merged.append(merge)

all_merged = pandas.concat(merged)
all_merged = all_merged.sort_values(['op'] + index)
all_merged.head()

###########################################
# Markdown
# ++++++++

piv = all_merged.pivot_table(values=['SpeedUp'], index=index, columns=['op'])
piv = piv.reset_index(drop=False).sort_values(index)
piv.columns = index + [_[1] for _ in piv.columns[4:]]
#print(piv.to_markdown(index=False, floatfmt=".2f"))

####################################
# Graphs
# ++++++

df = all_merged
graph_col = ['op', 'axes']
set_val = set(tuple(_[1:]) for _ in df[graph_col].itertuples())

axes = list(sorted(set(df['axes'])))
op = list(sorted(set(df['op'])))

for suffix in ['_ref', '_new']:
    fig, ax = plt.subplots(len(axes), len(op), figsize=(14, 14))
    for i, a in enumerate(axes):
        for j, o in enumerate(op):
            sub = df[(df['op'] == o) & (df['axes'] == a)]
            piv = sub.pivot("N", "fct", "average" + suffix)
            ref = piv['numpy'].copy()
            for c in piv.columns:
                piv[c] = ref / piv[c]
            piv.plot(ax=ax[i, j], logx=True)
            shape = list(sub['shape'])[0]
            ax[i, j].set_title("%s - %s - %s" % (o, a, shape), fontsize=5)
            ax[i, j].legend(fontsize=5)
            plt.setp(ax[i, j].get_xticklabels(), fontsize=5)
            plt.setp(ax[i, j].get_yticklabels(), fontsize=5)
    fig.suptitle(suffix)

plt.show()
