# -*- coding: utf-8 -*-
"""
@brief      test log(time=285s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
from pyquickhelper.pycode import (
    add_missing_development_version, ExtTestCase
)
import skl2onnx
import mlprodict


class TestNotebookTopk(ExtTestCase):

    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    def test_notebook_topk_cpp(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertNotEmpty(mlprodict is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        test_notebook_execution_coverage(__file__, "topk_cpp", folder,
                                         this_module_name="mlprodict", fLOG=fLOG)

    def test_notebook_topk_onnx(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertNotEmpty(mlprodict is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        test_notebook_execution_coverage(__file__, "onnx_topk", folder,
                                         this_module_name="mlprodict", fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
