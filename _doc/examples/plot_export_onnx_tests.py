"""
.. _l-export-onnx-test:

Walk through all methods to export an ONNX model
================================================

An ONNX model can be exported into many formats
(see :ref:`l-api-export-onnx`). This example checks the
availibility through all onnx examples and all formats.

.. contents::
    :local:

"""
import os
import numpy
from pandas import DataFrame
import matplotlib.pyplot as plt
from tqdm import tqdm
from mlprodict.testing.onnx_backend import enumerate_onnx_tests
from mlprodict.onnx_tools.onnx_export import (
    export2onnx, export2tf2onnx, export2xop,
    export2python, export2numpy, export2cpp)

#####################################
# Load the tests
# ++++++++++++++

tests = []
for test in tqdm(enumerate_onnx_tests('node')):
    tests.append(test)

#####################################
# Code
# ++++

conv = dict(onnx=export2onnx,
            tf2onnx=export2tf2onnx,
            xop=export2xop,
            python=export2python,
            numpy=export2numpy,
            cpp=export2cpp)

for fmt in conv:
    if not os.path.exists(fmt):
        os.mkdir(fmt)


data = []
for test in tqdm(tests):
    for fmt, fct in conv.items():
        onx = test.onnx_model
        ext = ".cpp" if 'cpp' in fmt else ".py"
        try:
            code = fct(onx)
            error = ""
        except Exception as e:
            error = str(e)
            code = None
        obs = dict(name=test.name, format=fmt, error=error,
                   ok=1 if error == "" else 0, code=code)
        data.append(obs)
        if code is not None:
            filename = os.path.join(fmt, test.name + ext)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code)


#####################################
# Status and summary
# ++++++++++++++++++

df = DataFrame(data)
summary = df.pivot("name", "format", "ok").mean(axis=0).T
print(summary)


#####################################
# Graph
# +++++

summary.plot.bar(title="Conversion coverage")


#####################################
# Errors
# ++++++

for obs in data:
    if obs['error'] != '':
        print(f"{obs['name']} | {obs['format']} | {obs['error']}")


# plt.show()
