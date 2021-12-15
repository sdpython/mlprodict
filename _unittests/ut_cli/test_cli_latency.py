"""
@brief      test tree node (time=4s)
"""
import os
import unittest
import pickle
import pandas
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
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LinearRegression()
        clr.fit(X_train, y_train)
        onx = to_onnx(clr, X[:1], black_op={'LinearRegressor'})
        with open(outonnx, "wb") as f:
            f.write(onx.SerializeToString())

        st = BufferedPrint()
        res = latency(outonnx)
        expected = ['average', 'context_size', 'deviation', 'max_exec', 'min_exec',
                    'number', 'repeat', 'ttime']
        self.assertEqual(list(sorted(res)), expected)

        st = BufferedPrint()
        res = latency(outonnx, max_time=0.5)
        self.assertEqual(list(sorted(res)), expected)
        self.assertGreater(res['ttime'], 0.5)


if __name__ == "__main__":
    unittest.main()
