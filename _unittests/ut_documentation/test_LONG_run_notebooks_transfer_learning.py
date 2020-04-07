# -*- coding: utf-8 -*-
"""
@brief      test log(time=15s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
from pyquickhelper.pycode import (
    add_missing_development_version, ExtTestCase
)
import mlprodict


class TestNotebookTransferLearning(ExtTestCase):

    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    def test_notebook_topk(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertNotEmpty(mlprodict is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        to_copy = ["800px-Tour_Eiffel_Wikimedia_Commons_(cropped).jpg"]
        test_notebook_execution_coverage(__file__, "transfer_learning", folder,
                                         this_module_name="mlprodict", fLOG=fLOG,
                                         copy_files=to_copy)


if __name__ == "__main__":
    unittest.main()
