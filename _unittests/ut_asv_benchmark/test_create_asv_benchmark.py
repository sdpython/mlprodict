"""
@brief      test log(time=2s)
"""
import os
import unittest
from logging import getLogger
from io import StringIO
import numpy
import pandas
import skl2onnx
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.asv_benchmark import create_asv_benchmark


class TestCreateAsvBenchmark(ExtTestCase):

    def test_create_asv_benchmark(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_create_asv_benchmark")
        created = create_asv_benchmark(
            location=temp, models={'LogisticRegression', 'LinearRegression'},
            verbose=5, fLOG=fLOG)
        print(created)
        self.assertGreater(len(created), 2)


if __name__ == "__main__":
    unittest.main()
