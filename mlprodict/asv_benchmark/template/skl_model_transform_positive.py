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
from sklearn.decomposition import NMF

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkTransformPositive
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611


class TemplateBenchmarkTransformPositive(_CommonAsvSklBenchmarkTransformPositive):
    """
    :epkg:`asv` example for a transform,
    Full template can be found in
    `common_asv_skl.py <https://github.com/sdpython/mlprodict/blob/
    master/mlprodict/asv_benchmark/common_asv_skl.py>`_.
    """
    params = [
        ['skl', 'pyrt', 'ort'],  # values for runtime
        [1, 100, 10000],  # values for N
        [4, 20],  # values for nf
        [9, 10, 11],  # values for opset
        ['float', 'double'],  # values for dtype
        [None],  # values for optim
    ]

    # additional parameters

    def setup_cache(self):  # pylint: disable=W0235
        super().setup_cache()

    def _create_model(self):
        return NMF()
