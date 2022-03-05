"""
@brief      test log(time=81s)
"""
import unittest
import os
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.script_testing import verify_script


class TestSklearnExample(ExtTestCase):

    def test_plot_iris_logistic(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        plot = os.path.join(data, "plot_iris_logistic.py")
        res = verify_script(plot)
        self.assertIsInstance(res, dict)
        loc = res['locals']
        self.assertIn('X', loc)
        self.assertIn('logreg', loc)
        self.assertIn('logreg_onnx', loc)

    def test_plot_anomaly_comparison(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        plot = os.path.join(data, "plot_anomaly_comparison.py")
        res = verify_script(plot, try_onnx=False)
        self.assertIsInstance(res, dict)
        loc = res['locals']
        self.assertIn('algorithm', loc)

    def test_plot_isotonic_regression(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        plot = os.path.join(data, "plot_isotonic_regression.py")
        try:
            verify_script(plot, try_onnx=False)
        except NameError as e:
            # Issues with local variable in comprehension list.
            self.assertIn("'y'", str(e))

    def test_plot_kernel_ridge_regression(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        plot = os.path.join(data, "plot_kernel_ridge_regression.py")
        res = verify_script(plot, try_onnx=False)
        self.assertIsInstance(res, dict)
        loc = res['locals']
        self.assertNotEmpty(filter(lambda n: n.endswith('_onnx'), loc))


if __name__ == "__main__":
    # TestSklearnExample().test_plot_kernel_ridge_regression()
    unittest.main()
