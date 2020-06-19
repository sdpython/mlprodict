"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.tools.asv_options_helper import version2number


class TestCreateAsvBenchmarkHelper(ExtTestCase):

    def test_version2number(self):
        for v in ['0.23.1', '0.24.dev0', '1.5.107']:
            r = version2number(v)
            self.assertGreater(r, 0.02)
            self.assertLess(r, 2)


if __name__ == "__main__":
    unittest.main()
