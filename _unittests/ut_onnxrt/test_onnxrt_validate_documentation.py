"""
@brief      test log(time=40s)
"""
import unittest
from logging import getLogger
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper.version_helper import compare_module_version
import skl2onnx
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, sklearn_operators
from mlprodict.onnxrt.validate import sklearn__all__
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.doc_write_helper import enumerate_visual_onnx_representation_into_rst


class TestOnnxrtValidateDocumentation(ExtTestCase):

    def test_validate_sklearn_store_models(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        rows = list(enumerate_validated_operator_opsets(
            verbose=0, models={"LinearRegression"}, opset_min=10,
            store_models=True, fLOG=fLOG))

        self.assertNotEmpty(rows)
        self.assertIn('MODEL', rows[0])
        self.assertIn('ONNX', rows[0])
        self.assertIsInstance(rows[0]['MODEL'], LinearRegression)
        oinf = OnnxInference(rows[0]['ONNX'])
        dot = oinf.to_dot()
        self.assertIn('LinearRegressor', dot)

    @unittest.skipIf(compare_module_version(skl2onnx.__version__, "1.5.0") <= 0,
                     reason="some node have null names")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_write_documentation_converters(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        subs = []
        for sub in sorted(sklearn__all__):
            models = sklearn_operators(sub)
            if len(models) > 0:
                rows = []
                for row in enumerate_visual_onnx_representation_into_rst(sub):
                    self.assertIn("digraph", row)
                    rows.append(row)
                if len(rows) == 0:
                    continue
                rows = [".. _l-skl2onnx-%s:" % sub, "", "=" * len(sub),
                        sub, "=" * len(sub), "", ".. contents::",
                        "    :local:", ""] + rows
                rows.append('')
                subs.append(sub)
                fLOG("subfolder '{}' - {} scenarios.".format(sub, len(models)))

        self.assertGreater(len(subs), 2)


if __name__ == "__main__":
    unittest.main()
