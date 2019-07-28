# -*- coding: utf-8 -*-
"""
@brief      test log(time=20s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
from pyquickhelper.pycode import add_missing_development_version, ExtTestCase
import mlprodict


class TestFunctionTestNotebookGrammar(ExtTestCase):

    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    def test_notebook_sklearn_grammar(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertNotEmpty(mlprodict is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        test_notebook_execution_coverage(__file__, "sklearn_grammar", folder,
                                         this_module_name="mlprodict", fLOG=fLOG,
                                         copy_files=["README.txt"])


if __name__ == "__main__":
    unittest.main()
