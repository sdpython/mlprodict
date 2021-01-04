"""
@brief      test log(time=3s)
"""

import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools import measure_time


class TestMeasureTime(ExtTestCase):

    def test_vector_count(self):
        def fct():
            X = numpy.ones((1000, 5))
            return X
        res = measure_time(
            "fct", context={"fct": fct}, div_by_number=False, number=100)
        self.assertIn("average", res)
        res = measure_time(
            "fct", context={"fct": fct}, div_by_number=True, number=100)
        self.assertIn("average", res)
        res = measure_time(
            "fct", context={"fct": fct}, div_by_number=True, number=1000)
        self.assertIn("average", res)


if __name__ == "__main__":
    unittest.main()
