"""
@brief      test log(time=2s)
"""
import unittest
import os
import numpy
import onnx
from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=E0611
    Fail as OrtFail, InvalidArgument as OrtInvalidArgument)
from pyquickhelper.pycode import (
    ExtTestCase, skipif_appveyor, skipif_circleci,
    skipif_travis, skipif_azure)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestBugsOnnxrtOnnxConverter(ExtTestCase):

    @skipif_appveyor("old version of onnxconvert-common")
    @skipif_circleci("old version of onnxconvert-common")
    @skipif_travis("old version of onnxconvert-common")
    @skipif_azure("old version of onnxconvert-common")
    def test_bug_apply_clip(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        y_train = y_train.astype(numpy.float32)
        clr = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=3)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                            target_opset=12)

        oinf2 = OnnxInference(model_def, runtime='python_compiled')
        res = oinf2.run({'X': X_test[:5]})
        self.assertGreater(len(res), 1)

    def fx_train(self, runtime):
        data = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            "data", "fw_train_LinearRegression.onnx")
        with open(data, 'rb') as f:
            model = onnx.load(f)
        for node in list(model.graph.node):  # pylint: disable=E1101
            if node.name == '':
                node.name = '%s_%d' % (node.op_type, id(node))
            for i in range(len(node.output)):  # pylint: disable=C0200
                if node.output[i] == '':
                    node.output[i] = "%s:%d" % (node.name, i)
        # with open('debug.onnx', 'wb') as f:
        #     f.write(model.SerializeToString())
        oinf = OnnxInference(model, runtime=runtime)
        grad = numpy.random.randn(25, 1).astype(numpy.float32).T
        X = numpy.random.randn(25, 10).astype(numpy.float32)
        coef = numpy.random.randn(10).astype(numpy.float32).reshape((10, 1))
        intercept = numpy.random.randn(1).astype(numpy.float32).reshape((1, ))
        res = oinf.run({'X': X, 'coef': coef, 'intercept': intercept},
                       yield_ops={'variable_grad': grad})
        self.assertEqual(res['X_grad'].shape, X.shape)
        self.assertEqual(res['coef_grad'].shape, coef.shape)
        self.assertEqual(res['intercept_grad'].shape, intercept.shape)

    def test_fx_train(self):
        for rt in ['python', 'python_compiled',
                   'onnxruntime1', 'onnxruntime2']:
            with self.subTest(runtime=rt):
                if rt == 'python_compiled':
                    self.assertRaise(
                        lambda rt=rt: self.fx_train(rt), RuntimeError)
                elif rt == 'python':
                    self.fx_train(rt)
                elif rt == 'onnxruntime1':
                    self.assertRaise(
                        lambda rt=rt: self.fx_train(rt), (RuntimeError, OrtFail))
                elif rt == 'onnxruntime2':
                    self.assertRaise(
                        lambda rt=rt: self.fx_train(rt), RuntimeError)
                else:
                    raise ValueError(f"Unexpected runtime {rt!r}.")

    def fx_train_cls(self, runtime):
        data = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            "data", "fw_train_LogisticRegression.onnx")
        with open(data, 'rb') as f:
            model = onnx.load(f)
        for node in list(model.graph.node):  # pylint: disable=E1101
            if node.name == '':
                node.name = '%s_%d' % (node.op_type, id(node))
            for i in range(len(node.output)):  # pylint: disable=C0200
                if node.output[i] == '':
                    node.output[i] = "%s:%d" % (node.name, i)
        # with open('debug.onnx', 'wb') as f:
        #     f.write(model.SerializeToString())
        oinf = OnnxInference(model, runtime=runtime)
        grad = numpy.random.randn(25, 3).astype(numpy.float32)
        X = numpy.random.randn(25, 10).astype(numpy.float32)
        coef = numpy.random.randn(10, 3).astype(numpy.float32)
        intercept = numpy.random.randn(3).astype(numpy.float32)
        res = oinf.run({'X': X, 'coef': coef, 'intercept': intercept},
                       yield_ops={'probabilities_grad': grad})
        self.assertEqual(res['X_grad'].shape, X.shape)
        self.assertEqual(res['coef_grad'].shape, coef.shape)
        self.assertEqual(res['intercept_grad'].shape, intercept.shape)

    def test_fx_train_cls(self):
        for rt in ['python', 'python_compiled',
                   'onnxruntime1', 'onnxruntime2']:
            with self.subTest(runtime=rt):
                if rt == 'python_compiled':
                    self.assertRaise(
                        lambda rt=rt: self.fx_train_cls(rt), RuntimeError)
                elif rt == 'python':
                    self.fx_train_cls(rt)
                elif rt == 'onnxruntime1':
                    self.assertRaise(
                        lambda rt=rt: self.fx_train_cls(rt),
                        (RuntimeError, OrtFail, OrtInvalidArgument))
                elif rt == 'onnxruntime2':
                    self.assertRaise(
                        lambda rt=rt: self.fx_train_cls(rt), RuntimeError)
                else:
                    raise ValueError(f"Unexpected runtime {rt!r}.")


if __name__ == "__main__":
    # TestBugsOnnxrtOnnxConverter().test_fx_train_cls()
    unittest.main()
