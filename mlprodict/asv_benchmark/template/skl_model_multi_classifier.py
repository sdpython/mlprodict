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
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx
# Import specific to this model.
from sklearn.tree import DecisionTreeClassifier  # pylint: disable=C0411

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkMultiClassifier  # pylint: disable=C0412
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611, C0412
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611, C0412


class TemplateBenchmarkMultiClassifier(_CommonAsvSklBenchmarkMultiClassifier):
    """
    :epkg:`asv` example for a classifier,
    Full template can be found in
    `common_asv_skl.py <https://github.com/sdpython/mlprodict/
    blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_.
    """
    params = [
        ['skl', 'pyrtc', 'ort'],  # values for runtime
        [1, 10, 100, 1000, 10000],  # values for N
        [4, 20],  # values for nf
        [get_opset_number_from_onnx()],  # values for opset
        ['float', 'double'],  # values for dtype
        [None],  # values for optim
    ]

    # additional parameters

    def setup_cache(self):  # pylint: disable=W0235
        super().setup_cache()

    def _create_model(self):
        return DecisionTreeClassifier()
