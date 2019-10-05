# Import specific to this model.
from sklearn.linear_model import LogisticRegression

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkClassifier
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611


class LogisticRegression_m_cl_liblinear_solverliblinear_11Classifier(_CommonAsvSklBenchmarkClassifier):
    "asv test for a classifier"
    # Full template can be found in
    # https://github.com/sdpython/mlprodict/blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_

    params = [
        ['skl', 'pyrt'],
        [1, 100, 10000],
        [4, 20],
    ]
    param_names = ['rt', 'N', 'nf']
    target_opset = 11

    def _create_model(self):
        return LogisticRegression(
            solver='liblinear'
        )
