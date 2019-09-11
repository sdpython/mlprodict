"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report


class TestRtValidateVotingClassifier(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianMixture_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = False
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"VotingClassifier"}, opset_min=9,
            opset_max=11, fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, p: 'm-cl' in p))
        self.assertGreater(len(rows), 1)
        self.assertIn('skl_nop', rows[0])
        self.assertIn('onx_size', rows[-1])
        piv = summary_report(DataFrame(rows))
        self.assertEqual(piv.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
