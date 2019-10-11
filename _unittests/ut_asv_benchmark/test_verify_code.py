"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.asv_benchmark.verify_code import verify_code, ImperfectPythonCode

source = '''
import numpy  # pylint: disable=W0611
# Import specific to this model.
from sklearn.ensemble import VotingClassifier

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkClassifier
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611


class VotingClassifier_b_cl_logreg_noflatten_votingsoft_flatten_transfda804c_9_benchClassifier(_CommonAsvSklBenchmarkClassifier):
    """
    :epkg:`asv` test for a classifier,
    Full template can be found in
    `common_asv_skl.py <https://github.com/sdpython/mlprodict/
    blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_.
    """

    params = [
        ['skl', 'pyrt', 'ort'],
        (1, 100, 10000),
        (4, 20),
    ]
    param_names = ['rt', 'N', 'nf']
    target_opset = 9

    def setup_cache(self):
        super().setup_cache()

    def _create_model(self):
        return VotingClassifier(
            estimators=[('lr1', LogisticRegression(C=1.0, class_weight=None,
            dual=False, fit_intercept=True,
            intercept_scaling=1, l1_ratio=None, max_iter=100,
            multi_class='auto', n_jobs=None, penalty='l2',
            random_state=None, solver='liblinear', tol=0.0001, verbose=0,
            warm_start=False)), ('lr2', LogisticRegression(C=1.0,
            class_weight=None, dual=False, fit_intercept=False,
            intercept_scaling=1, l1_ratio=None, max_iter=100,
            multi_class='auto', n_jobs=None, penalty='l2',
            random_state=None, solver='liblinear', tol=0.0001, verbose=0,
            warm_start=False))], flatten_transform=False, voting='soft'
        )
'''


class TestVerifyCode(ExtTestCase):

    def test_verify_code(self):
        self.assertRaise(lambda: verify_code(source), ImperfectPythonCode)


if __name__ == "__main__":
    unittest.main()
