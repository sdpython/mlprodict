# -*- coding: utf-8 -*-
"""
@brief      test log(time=120s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
from pyquickhelper.pycode import (
    add_missing_development_version, ExtTestCase,
    skipif_appveyor, skipif_circleci, skipif_azure)
import mlprodict


class TestNotebookOnnxOperatorCost(ExtTestCase):

    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    @skipif_appveyor("too long")
    @skipif_circleci("too long")
    @skipif_azure("too long")
    def test_notebook_onnx_pdist(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertNotEmpty(mlprodict is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        test_notebook_execution_coverage(__file__, "onnx_operator_cost", folder,
                                         this_module_name="mlprodict", fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
