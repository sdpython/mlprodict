"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
from io import StringIO
import numpy
import pandas
import skl2onnx
from pyquickhelper.pycode import ExtTestCase
from mlprodict.asv_benchmark.template.skl_model import TemplateBenchmark


class TestAsvTemplateBenchmark(ExtTestCase):

    def test_template_benchmark(self):
        cl = TemplateBenchmark()
        cl.setup()
        for method in cl.__class__.__dict__:
            if method.split('_')[0] in ('time', 'peakmem'):
                meth = getattr(cl.__class__, method)
                meth(cl)


if __name__ == "__main__":
    unittest.main()
