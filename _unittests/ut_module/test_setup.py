"""
@brief      test tree node (time=50s)
"""
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pyquickhelper.pycode import ExtTestCase
from python3_module_template import check, _setup_hook


class TestSetup(ExtTestCase):

    def test_check(self):
        self.assertTrue(check())

    def test_setup_hook(self):
        _setup_hook()

    def test_setup_hook_print(self):
        with redirect_stdout(StringIO()):
            _setup_hook(True)


if __name__ == "__main__":
    unittest.main()
