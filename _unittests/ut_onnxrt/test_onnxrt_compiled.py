"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import timeit
from io import BytesIO
import pickle
import numpy
from pyquickhelper.pycode import ExtTestCase, skipif_circleci
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from skl2onnx import to_onnx
from skl2onnx.algebra.onnx_ops import OnnxAdd  # pylint: disable=E0611
from mlprodict.onnxrt import OnnxInference
from mlprodict import __max_supported_opset__ as TARGET_OPSET


class TestOnnxrtCompiled(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_idi(self):
        idi = numpy.identity(2).astype(numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})

        oinf = OnnxInference(model_def, runtime="python_compiled")
        res = oinf.run({'X': idi.astype(numpy.float32)})
        self.assertEqual(idi * 2, res['Y'])
        self.assertIn('_run_compiled', oinf.__dict__)
        self.assertIn('_run_compiled_code', oinf.__dict__)
        code = oinf._run_compiled_code  # pylint: disable=W0212,E1101
        self.assertIsInstance(code, str)
        self.assertIn('def compiled_run(dict_inputs, yield_ops=None):', code)
        self.assertIn('(Y, ) = n0_add(X, Ad_Addcst)', code)
        self.assertIn(
            ' def compiled_run(dict_inputs, yield_ops=None):', str(oinf))

    def test_onnxt_idi_debug(self):
        idi = numpy.identity(2).astype(numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})

        oinf = OnnxInference(model_def, runtime="python_compiled_debug")
        res, out, err = self.capture(
            lambda: oinf.run({'X': idi.astype(numpy.float32)}))
        self.assertEmpty(err)
        self.assertIn("-='i.X'", out)
        self.assertIn("-='o.Y'", out)
        self.assertEqual(idi * 2, res['Y'])
        self.assertIn('_run_compiled', oinf.__dict__)
        self.assertIn('_run_compiled_code', oinf.__dict__)
        code = oinf._run_compiled_code  # pylint: disable=W0212,E1101
        self.assertIsInstance(code, str)
        self.assertIn('def compiled_run(dict_inputs, yield_ops=None):', code)
        self.assertIn('(Y, ) = n0_add(X, Ad_Addcst)', code)
        self.assertIn(
            ' def compiled_run(dict_inputs, yield_ops=None):', str(oinf))

    @skipif_circleci('fails to finish')
    def test_onnxt_iris_adaboost_regressor_dt(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        y_train = y_train.astype(numpy.float32)
        clr = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=3),
            n_estimators=3)
        clr.fit(X_train, y_train)
        X_test = X_test.astype(numpy.float32)
        X_test = numpy.vstack([X_test[:3], X_test[-3:]])

        model_def = to_onnx(clr, X_train.astype(numpy.float32))

        oinf1 = OnnxInference(model_def, runtime='python')
        res1 = oinf1.run({'X': X_test})['variable']

        oinf2 = OnnxInference(model_def, runtime='python_compiled')
        res2 = oinf2.run({'X': X_test})['variable']

        self.assertEqualArray(res1, res2)

        X_test = X_test[:1]
        t1 = timeit.repeat(stmt="oinf1.run({'X': X_test})", setup='pass',
                           repeat=5, number=1000,
                           globals={'X_test': X_test, 'oinf1': oinf1})
        me1 = sum(t1) / len(t1)
        t2 = timeit.repeat(stmt="oinf2.run({'X': X_test})", setup='pass',
                           repeat=5, number=1000,
                           globals={'X_test': X_test, 'oinf2': oinf2})
        me2 = sum(t2) / len(t2)
        self.assertGreater(me1, me2)
        # print(me1, me2)
        # print(oinf2._run_compiled_code)
        self.assertIn(
            ' def compiled_run(dict_inputs, yield_ops=None):', str(oinf2))

    def test_onnxt_reduce_size(self):
        idi = numpy.identity(2).astype(numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})

        oinf = OnnxInference(model_def, runtime="python_compiled")
        res = oinf.run({'X': idi.astype(numpy.float32)})
        self.assertEqual(idi * 2, res['Y'])

        oinf.reduce_size(False)
        res = oinf.run({'X': idi.astype(numpy.float32)})
        self.assertEqual(idi * 2, res['Y'])
        st = BytesIO()
        try:
            pickle.dump(oinf, st)
        except AttributeError:
            # missing obj
            pass

        oinf = OnnxInference(model_def, runtime="python_compiled")
        res = oinf.run({'X': idi.astype(numpy.float32)})
        self.assertEqual(idi * 2, res['Y'])

        oinf.reduce_size(True)
        res = oinf.run({'X': idi.astype(numpy.float32)})
        self.assertEqual(idi * 2, res['Y'])
        st = BytesIO()
        pickle.dump(oinf, st)
        val = st.getvalue()
        oinf2 = pickle.load(BytesIO(val))
        self.assertNotEmpty(oinf2)


if __name__ == "__main__":
    unittest.main()
