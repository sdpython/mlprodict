"""
@brief      test log(time=2s)
"""
import os
import unittest
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx.common.exceptions import MissingShapeCalculator
from pyquickhelper.pycode import ExtTestCase

from mlprodict.asv_benchmark.template.skl_model_classifier import (
    TemplateBenchmarkClassifier
)
from mlprodict.asv_benchmark.template.skl_model_clustering import (
    TemplateBenchmarkClustering
)
from mlprodict.asv_benchmark.template.skl_model_multi_classifier import (
    TemplateBenchmarkMultiClassifier
)
from mlprodict.asv_benchmark.template.skl_model_regressor import (
    TemplateBenchmarkRegressor
)
from mlprodict.asv_benchmark.template.skl_model_outlier import (
    TemplateBenchmarkOutlier
)
from mlprodict.asv_benchmark.template.skl_model_trainable_transform import (
    TemplateBenchmarkTrainableTransform
)
from mlprodict.asv_benchmark.template.skl_model_transform import (
    TemplateBenchmarkTransform
)


class TestAsvTemplateBenchmark(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ))
    def test_template_benchmark_classifier(self):
        if not os.path.exists('_cache'):
            os.mkdir('_cache')
        cl = TemplateBenchmarkClassifier()
        res = {}
        cl.setup_cache()
        N = 60
        nf = cl.params[2][1]
        opset = 10
        dtype = 'float'
        optim = None
        for runtime in ['skl', 'pyrt', 'ort']:
            cl.setup(runtime, N, nf, opset, dtype, optim)
            self.assertEqual(cl.X.shape, (N, nf))
            for method in dir(cl):
                if method.split('_')[0] in ('time', 'peakmem', 'track'):
                    meth = getattr(cl.__class__, method)
                    res[method, runtime] = meth(
                        cl, runtime, N, nf, opset, dtype, optim)
        self.assertEqual(len(res), 18)
        exp = [('time_predict', 'skl'), ('peakmem_predict', 'skl'),
               ('track_score', 'skl'), ('track_onnxsize', 'skl'),
               ('time_predict', 'pyrt'), ('peakmem_predict', 'pyrt'),
               ('track_score', 'pyrt'), ('track_onnxsize', 'pyrt'),
               ('time_predict', 'ort'), ('peakmem_predict', 'ort'),
               ('track_score', 'ort'), ('track_onnxsize', 'ort'),
               ('track_nbnodes', 'skl'), ('track_nbnodes', 'ort'),
               ('track_opset', 'skl'), ('track_opset', 'pyrt'),
               ('track_nbnodes', 'pyrt'), ('track_opset', 'ort')]
        self.assertEqual(set(exp), set(res))

    @ignore_warnings(category=(UserWarning, ))
    def test_template_benchmark_clustering(self):
        if not os.path.exists('_cache'):
            os.mkdir('_cache')
        cl = TemplateBenchmarkClustering()
        res = {}
        cl.setup_cache()
        N = 60
        nf = cl.params[2][1]
        opset = 10
        dtype = 'float'
        optim = None
        for runtime in ['skl', 'pyrt']:
            cl.setup(runtime, N, nf, opset, dtype, optim)
            self.assertEqual(cl.X.shape, (N, nf))
            for method in dir(cl):
                if method.split('_')[0] in ('time', 'peakmem', 'track'):
                    meth = getattr(cl.__class__, method)
                    res[method, runtime] = meth(
                        cl, runtime, N, nf, opset, dtype, optim)
        self.assertEqual(len(res), 12)
        exp = [('time_predict', 'skl'), ('peakmem_predict', 'skl'),
               ('track_score', 'skl'), ('track_onnxsize', 'skl'),
               ('time_predict', 'pyrt'), ('peakmem_predict', 'pyrt'),
               ('track_score', 'pyrt'), ('track_onnxsize', 'pyrt'),
               ('track_nbnodes', 'skl'), ('track_opset', 'skl'),
               ('track_opset', 'pyrt'), ('track_nbnodes', 'pyrt')]
        self.assertEqual(set(exp), set(res))

    @ignore_warnings(category=(UserWarning, ))
    def test_template_benchmark_regressor(self):
        if not os.path.exists('_cache'):
            os.mkdir('_cache')
        cl = TemplateBenchmarkRegressor()
        res = {}
        cl.setup_cache()
        N = 60
        nf = cl.params[2][1]
        opset = 10
        dtype = 'float'
        optim = None
        for runtime in ['skl', 'pyrt', 'ort']:
            cl.setup(runtime, N, nf, opset, dtype, optim)
            self.assertEqual(cl.X.shape, (N, nf))
            for method in dir(cl):
                if method.split('_')[0] in ('time', 'peakmem', 'track'):
                    meth = getattr(cl.__class__, method)
                    res[method, runtime] = meth(
                        cl, runtime, N, nf, opset, dtype, optim)
        self.assertEqual(len(res), 18)
        exp = [('time_predict', 'skl'), ('peakmem_predict', 'skl'),
               ('track_score', 'skl'), ('track_onnxsize', 'skl'),
               ('time_predict', 'pyrt'), ('peakmem_predict', 'pyrt'),
               ('track_score', 'pyrt'), ('track_onnxsize', 'pyrt'),
               ('time_predict', 'ort'), ('peakmem_predict', 'ort'),
               ('track_score', 'ort'), ('track_onnxsize', 'ort'),
               ('track_nbnodes', 'skl'), ('track_nbnodes', 'ort'),
               ('track_opset', 'skl'), ('track_opset', 'pyrt'),
               ('track_nbnodes', 'pyrt'), ('track_opset', 'ort')]
        self.assertEqual(set(exp), set(res))

    @ignore_warnings(category=(UserWarning, ))
    def test_template_benchmark_multi_classifier(self):
        if not os.path.exists('_cache'):
            os.mkdir('_cache')
        cl = TemplateBenchmarkMultiClassifier()
        res = {}
        cl.setup_cache()
        N = 60
        nf = cl.params[2][1]
        opset = 10
        dtype = 'float'
        optim = None
        for runtime in ['skl', 'pyrt']:
            try:
                cl.setup(runtime, N, nf, opset, dtype, optim)
            except NotImplementedError:
                # not implemented
                return
            self.assertEqual(cl.X.shape, (N, nf))
            for method in dir(cl):
                if method.split('_')[0] in ('time', 'peakmem', 'track'):
                    meth = getattr(cl.__class__, method)
                    res[method, runtime] = meth(
                        cl, runtime, N, nf, opset, dtype, optim)
        self.assertEqual(len(res), 18)
        exp = [('time_predict', 'skl'), ('peakmem_predict', 'skl'),
               ('track_score', 'skl'), ('track_onnxsize', 'skl'),
               ('time_predict', 'pyrt'), ('peakmem_predict', 'pyrt'),
               ('track_score', 'pyrt'), ('track_onnxsize', 'pyrt'),
               ('track_nbnodes', 'skl'), ('track_opset', 'skl'),
               ('track_opset', 'pyrt'), ('track_nbnodes', 'pyrt')]
        self.assertEqual(set(exp), set(res))

    @ignore_warnings(category=(UserWarning, ))
    def test_template_benchmark_outlier(self):
        if not os.path.exists('_cache'):
            os.mkdir('_cache')
        cl = TemplateBenchmarkOutlier()
        res = {}
        cl.setup_cache()
        N = 60
        nf = cl.params[2][1]
        expect = 12
        opset = 10
        dtype = 'float'
        optim = None
        for runtime in ['skl', 'pyrt']:
            try:
                cl.setup(runtime, N, nf, opset, dtype, optim)
            except MissingShapeCalculator:
                # Converter not yet implemented.
                expect = 0
                continue
            self.assertEqual(cl.X.shape, (N, nf))
            for method in dir(cl):
                if method.split('_')[0] in ('time', 'peakmem', 'track'):
                    meth = getattr(cl.__class__, method)
                    res[method, runtime] = meth(
                        cl, runtime, N, nf, opset, dtype, optim)
        if expect == 0:
            return
        self.assertEqual(len(res), expect)
        exp = [('time_predict', 'skl'), ('peakmem_predict', 'skl'),
               ('track_score', 'skl'), ('track_onnxsize', 'skl'),
               ('time_predict', 'pyrt'), ('peakmem_predict', 'pyrt'),
               ('track_score', 'pyrt'), ('track_onnxsize', 'pyrt'),
               ('track_nbnodes', 'skl'), ('track_opset', 'skl'),
               ('track_opset', 'pyrt'), ('track_nbnodes', 'pyrt')]
        self.assertEqual(set(exp), set(res))

    @ignore_warnings(category=(UserWarning, ))
    def test_template_benchmark_trainable_transform(self):
        if not os.path.exists('_cache'):
            os.mkdir('_cache')
        cl = TemplateBenchmarkTrainableTransform()
        res = {}
        cl.setup_cache()
        N = 60
        nf = cl.params[2][1]
        opset = 10
        dtype = 'float'
        expect = 12
        optim = None
        for runtime in ['skl', 'pyrt']:
            try:
                cl.setup(runtime, N, nf, opset, dtype, optim)
            except MissingShapeCalculator:
                # Converter not yet implemented.
                expect = 0
                continue
            self.assertEqual(cl.X.shape, (N, nf))
            for method in dir(cl):
                if method.split('_')[0] in ('time', 'peakmem', 'track'):
                    meth = getattr(cl.__class__, method)
                    res[method, runtime] = meth(
                        cl, runtime, N, nf, opset, dtype, optim)
        if expect == 0:
            return
        self.assertEqual(len(res), expect)
        exp = [('time_predict', 'skl'), ('peakmem_predict', 'skl'),
               ('track_score', 'skl'), ('track_onnxsize', 'skl'),
               ('time_predict', 'pyrt'), ('peakmem_predict', 'pyrt'),
               ('track_score', 'pyrt'), ('track_onnxsize', 'pyrt'),
               ('track_nbnodes', 'skl'), ('track_opset', 'skl'),
               ('track_opset', 'pyrt'), ('track_nbnodes', 'pyrt')]
        self.assertEqual(set(exp), set(res))

    @ignore_warnings(category=(UserWarning, ))
    def test_template_benchmark_transform(self):
        if not os.path.exists('_cache'):
            os.mkdir('_cache')
        cl = TemplateBenchmarkTransform()
        res = {}
        cl.setup_cache()
        N = 60
        nf = cl.params[2][1]
        opset = 10
        dtype = 'float'
        expect = 12
        optim = None
        for runtime in ['skl', 'pyrt']:
            try:
                cl.setup(runtime, N, nf, opset, dtype, optim)
            except MissingShapeCalculator:
                # Converter not yet implemented.
                expect = 0
                continue
            self.assertEqual(cl.X.shape, (N, nf))
            for method in dir(cl):
                if method.split('_')[0] in ('time', 'peakmem', 'track'):
                    meth = getattr(cl.__class__, method)
                    res[method, runtime] = meth(
                        cl, runtime, N, nf, opset, dtype, optim)
        if expect == 0:
            return
        self.assertEqual(len(res), expect)
        exp = [('time_predict', 'skl'), ('peakmem_predict', 'skl'),
               ('track_score', 'skl'), ('track_onnxsize', 'skl'),
               ('time_predict', 'pyrt'), ('peakmem_predict', 'pyrt'),
               ('track_score', 'pyrt'), ('track_onnxsize', 'pyrt'),
               ('track_nbnodes', 'skl'), ('track_opset', 'skl'),
               ('track_opset', 'pyrt'), ('track_nbnodes', 'pyrt')]
        self.assertEqual(set(exp), set(res))


if __name__ == "__main__":
    unittest.main()
