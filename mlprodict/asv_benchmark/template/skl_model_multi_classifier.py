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
import numpy  # pylint: disable=W0611
# Import specific to this model.
from sklearn.ensemble import RandomForestClassifier

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkMultiClassifier
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611


class TemplateBenchmarkMultiClassifier(_CommonAsvSklBenchmarkMultiClassifier):
    """
    :epkg:`asv` example for a classifier,
    Full template can be found in
    `common_asv_skl.py <https://github.com/sdpython/mlprodict/
    blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_.
    """

    params = [
        ['skl', 'pyrt', 'ort'],  # values for runtime
        [1, 100, 10000],  # values for N
        [4, 20],  # values for nf
    ]
    param_names = ['rt', 'N', 'nf']
    # additional parameters

    def setup_cache(self):  # pylint: disable=W0235
        super().setup_cache()

    def _create_model(self):
        return RandomForestClassifier()
