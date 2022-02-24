"""
@brief      test tree node (time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict import check


class TestSetup(ExtTestCase):

    def test_check(self):
        self.assertTrue(check())


if __name__ == "__main__":
    unittest.main()
