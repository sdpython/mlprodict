"""
@brief      test tree node (time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing import check_is_almost_equal


class TestTesting(ExtTestCase):

    def test_check_is_almost_equal(self):
        l1 = numpy.array([1, 2])
        l2 = numpy.array([1, 2])
        check_is_almost_equal(l1, l2)
        l1 = 3
        l2 = numpy.array([1, 2])
        self.assertRaise(lambda: check_is_almost_equal(l1, l2), TypeError)
        l1 = numpy.array([1, 3])
        l2 = numpy.array([1, 2])
        self.assertRaise(lambda: check_is_almost_equal(l1, l2), AssertionError)


if __name__ == "__main__":
    unittest.main()
