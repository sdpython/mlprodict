"""
A template to benchmark a model
with :epkg:`asv`. The benchmark can be run through
file `run_asv.sh <https://github.com/sdpython/mlprodict/blob/master/run_asv.sh>`_
on Linux or `run_asv.bat
<https://github.com/sdpython/mlprodict/blob/master/run_asv.bat>`_ on
Windows.

.. warning::
    On Windows, you should avoid cloning the repository
    on a folder with a long full name. Visual Studio tends to
    abide by the rule of the maximum path length even though
    the system is told otherwise.
"""
import os
from logging import getLogger
import numpy
import pickle

# Import specific to this model.
from sklearn.linear_model import LinearRegression

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkRegressor
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TemplateBenchmarkRegressor(_CommonAsvSklBenchmarkRegressor):
    # Full template can be found in
    # https://github.com/sdpython/mlprodict/blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_

    params = [
        ['skl', 'pyrt', 'ort'],  # values for runtime
        [1, 100, 10000],  # values for N
        [4, 20],  # values for nf
    ]
    param_names = ['rt', 'N', 'nf']

    def _create_model(self):
        return LinearRegression()
