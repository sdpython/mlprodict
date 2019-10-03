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
from pyquickhelper.pycode import ExtTestCase
from mlprodict.asv_benchmark.template.skl_model_classifier import (
    TemplateBenchmarkClassifier
)
from mlprodict.asv_benchmark.template.skl_model_regressor import (
    TemplateBenchmarkRegressor
)


class TestAsvTemplateBenchmark(ExtTestCase):

    def test_template_benchmark_classifier(self):
        if not os.path.exists('_cache'):
            os.mkdir('_cache')
        cl = TemplateBenchmarkClassifier()
        res = {}
        cl.setup_cache()
        N = 60
        nf = cl.params[2][1]
        for runtime in ['skl', 'pyrt', 'ort']:
            cl.setup(runtime, N, nf)
            self.assertEqual(cl.X.shape, (N, nf))
            for method in dir(cl):
                if method.split('_')[0] in ('time', 'peakmem', 'track'):
                    meth = getattr(cl.__class__, method)
                    res[method, runtime] = meth(cl, runtime, N, nf)
        self.assertEqual(len(res), 12)
        exp = [('time_predict', 'skl'), ('peakmem_predict', 'skl'),
               ('track_score', 'skl'), ('track_onnxsize', 'skl'),
               ('time_predict', 'pyrt'), ('peakmem_predict', 'pyrt'),
               ('track_score', 'pyrt'), ('track_onnxsize', 'pyrt'),
               ('time_predict', 'ort'), ('peakmem_predict', 'ort'),
               ('track_score', 'ort'), ('track_onnxsize', 'ort')]
        self.assertEqual(set(exp), set(res))

    def test_template_benchmark_regressor(self):
        if not os.path.exists('_cache'):
            os.mkdir('_cache')
        cl = TemplateBenchmarkRegressor()
        res = {}
        cl.setup_cache()
        N = 60
        nf = cl.params[2][1]
        for runtime in ['skl', 'pyrt', 'ort']:
            cl.setup(runtime, N, nf)
            self.assertEqual(cl.X.shape, (N, nf))
            for method in dir(cl):
                if method.split('_')[0] in ('time', 'peakmem', 'track'):
                    meth = getattr(cl.__class__, method)
                    res[method, runtime] = meth(cl, runtime, N, nf)
        self.assertEqual(len(res), 12)
        exp = [('time_predict', 'skl'), ('peakmem_predict', 'skl'),
               ('track_score', 'skl'), ('track_onnxsize', 'skl'),
               ('time_predict', 'pyrt'), ('peakmem_predict', 'pyrt'),
               ('track_score', 'pyrt'), ('track_onnxsize', 'pyrt'),
               ('time_predict', 'ort'), ('peakmem_predict', 'ort'),
               ('track_score', 'ort'), ('track_onnxsize', 'ort')]
        self.assertEqual(set(exp), set(res))


if __name__ == "__main__":
    unittest.main()
