"""
@brief      test tree node (time=10s)
"""
import os
import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder, ignore_warnings
from mlprodict.onnx_conv import to_onnx
from mlprodict.__main__ import main
from mlprodict.cli import latency


class TestCliLatency(ExtTestCase):

    def test_cli_latency(self):
        st = BufferedPrint()
        main(args=["latency", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("latency", res)

    @ignore_warnings(ConvergenceWarning)
    def test_latency_linreg(self):
        temp = get_temp_folder(__file__, "temp_latency")
        outonnx = os.path.join(temp, 'outolr.onnx')

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LinearRegression()
        clr.fit(X_train, y_train)
        onx = to_onnx(clr, X[:1], black_op={'LinearRegressor'})
        with open(outonnx, "wb") as f:
            f.write(onx.SerializeToString())

        res = latency(outonnx)
        expected = ['average', 'context_size', 'deviation', 'max_exec', 'min_exec',
                    'number', 'repeat', 'shape(X)', 'ttime']
        self.assertEqual(list(sorted(res)), expected)

        res = latency(outonnx, max_time=0.5)
        self.assertEqual(list(sorted(res)), expected)
        self.assertGreater(res['ttime'], 0.5)

        res = latency(outonnx, max_time=0.5, fmt='csv')
        self.assertIn('average,deviation', res)
        self.assertRaise(lambda: latency(outonnx, device="RR"), ValueError)
        self.assertRaise(lambda: latency(outonnx, device="R,R"), ValueError)

    @ignore_warnings(ConvergenceWarning)
    def test_latency_linreg_profile(self):
        temp = get_temp_folder(__file__, "temp_latency_profile")
        outonnx = os.path.join(temp, 'outolr.onnx')

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LinearRegression()
        clr.fit(X_train, y_train)
        onx = to_onnx(clr, X[:1], black_op={'LinearRegressor'})
        with open(outonnx, "wb") as f:
            f.write(onx.SerializeToString())

        for runtime in ('onnxruntime', 'onnxruntime1'):
            for prof in ('name', 'type'):
                with self.subTest(runtime=runtime, prof=prof):
                    o = os.path.join(temp, f'prof_{runtime}_{prof}.csv')
                    res = latency(outonnx, max_time=0.5, fmt='csv',
                                  profiling=prof, runtime=runtime,
                                  profile_output=o)
                    self.assertIn('average,deviation', res)
                    self.assertExists(o)


if __name__ == "__main__":
    unittest.main()
