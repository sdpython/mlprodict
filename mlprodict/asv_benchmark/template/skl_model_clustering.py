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
from onnx.defs import onnx_opset_version
# Import specific to this model.
from sklearn.cluster import KMeans

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkClustering
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611


class TemplateBenchmarkClustering(_CommonAsvSklBenchmarkClustering):
    """
    :epkg:`asv` example for a clustering algorithm,
    Full template can be found in
    `common_asv_skl.py <https://github.com/sdpython/mlprodict/
    blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_.
    """
    params = [
        ['skl', 'pyrt', 'ort'],  # values for runtime
        [1, 10, 100, 1000, 10000, 100000],  # values for N
        [4, 20],  # values for nf
        [onnx_opset_version()],  # values for opset
        ['float', 'double'],  # values for dtype
        [None],  # values for optim
    ]

    # additional parameters

    def setup_cache(self):  # pylint: disable=W0235
        super().setup_cache()

    def _create_model(self):
        return KMeans(n_clusters=2)
