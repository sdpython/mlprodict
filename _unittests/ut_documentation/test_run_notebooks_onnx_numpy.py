# -*- coding: utf-8 -*-
"""
@brief      test log(time=30s)
"""
import os
import unittest
from onnxruntime import __version__ as ort_version
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from pyquickhelper.loghelper import fLOG
from pyquickhelper.texthelper.version_helper import compare_module_version
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
from pyquickhelper.pycode import (
    add_missing_development_version, ExtTestCase
)
from skl2onnx import __version__ as skl2onnx_version
import mlprodict


class TestNotebookNumpyOnnx(ExtTestCase):

    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @unittest.skipIf(compare_module_version(ort_version, "0.4.0") <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_notebook_numpy_onnx(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertNotEmpty(mlprodict is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        test_notebook_execution_coverage(__file__, "numpy_api_onnx", folder,
                                         this_module_name="mlprodict", fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
