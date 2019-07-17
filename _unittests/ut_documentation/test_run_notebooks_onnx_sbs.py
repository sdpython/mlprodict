# -*- coding: utf-8 -*-
"""
@brief      test log(time=33s)
"""
import os
import unittest
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from pyquickhelper.loghelper import fLOG
from pyquickhelper.texthelper.version_helper import compare_module_version
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
from pyquickhelper.pycode import add_missing_development_version, ExtTestCase
from skl2onnx import __version__ as skl2onnx_version
from onnxruntime import __version__ as ort_version
import mlprodict


class TestFunctionTestNotebookOnnxSbs(ExtTestCase):

    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @unittest.skipIf(compare_module_version(ort_version, "0.4.0") <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_notebook_onnx_sbs(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertNotEmpty(mlprodict is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        test_notebook_execution_coverage(__file__, "onnx_sbs", folder,
                                         this_module_name="mlprodict", fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
