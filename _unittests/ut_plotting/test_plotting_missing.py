# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.plotting.plotting_validate_graph import _model_name


class TestPlottingMissing(ExtTestCase):

    def test_model_name(self):
        self.assertEqual('Nu', _model_name('Nu'))
        self.assertEqual('Select', _model_name('Select'))


if __name__ == "__main__":
    unittest.main()
