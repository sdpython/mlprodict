import numpy  # pylint: disable=W0611
# Import specific to this model.
from mlprodict.onnxrt.optim import onnx_optimisations
from sklearn.linear_model import LogisticRegression

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkClassifier
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611


class LogisticRegression_b_cl_64_liblinear_solverliblinear_onnx_10_benchClassifier(_CommonAsvSklBenchmarkClassifier):
    """
    :epkg:`asv` test for a classifier,
    Full template can be found in
    `common_asv_skl.py <https://github.com/sdpython/mlprodict/
    blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_.
    """

    params = [
        ['skl', 'pyrt'],
        (1, 100, 10000),
        (4, 20),
    ]
    param_names = ['rt', 'N', 'nf']
    xtest_dtype = numpy.float64
    target_opset = 10

    def setup_cache(self):
        super().setup_cache()

    def _create_model(self):
        return LogisticRegression(
            solver='liblinear'
        )


    def _optimize_onnx(self, onx):
        return onnx_optimisations(onx)
