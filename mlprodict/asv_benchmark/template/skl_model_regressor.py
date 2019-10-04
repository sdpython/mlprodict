"""
A template to benchmark a model
with :epkg:`asv`. The benchmark can be run through
file :epkg:`run_asv.sh` on Linux or :epkg:`run_asv.bat` on
Windows.

.. warning::
    On Windows, you should avoid cloning the repository
    on a folder with a long full name. Visual Studio tends to
    abide by the rule of the maximum path length even though
    the system is told otherwise.
"""
# Import specific to this model.
from sklearn.linear_model import LinearRegression

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkRegressor
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611


class TemplateBenchmarkRegressor(_CommonAsvSklBenchmarkRegressor):
    "asv example for a regressor"
    # Full template can be found in
    # https://github.com/sdpython/mlprodict/blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_

    params = [
        ['skl', 'pyrt', 'ort'],  # values for runtime
        [1, 100, 10000],  # values for N
        [4, 20],  # values for nf
    ]
    param_names = ['rt', 'N', 'nf']
    # additional parameters

    def _create_model(self):
        return LinearRegression()
