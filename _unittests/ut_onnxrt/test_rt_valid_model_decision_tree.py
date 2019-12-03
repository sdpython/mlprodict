"""
@brief      test log(time=9s)
"""
import unittest
from logging import getLogger
from onnx.defs import onnx_opset_version
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
import skl2onnx
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestRtValidateDecisionTree(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_DecisionTreeRegressor_onnxruntime(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = False
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"DecisionTreeRegressor"},
            opset_min=onnx_opset_version(), fLOG=myprint,
            runtime='onnxruntime2', debug=debug,
            filter_exp=lambda m, p: "-64" not in p))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_DecisionTreeRegressor_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = False
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"DecisionTreeRegressor"},
            opset_min=onnx_opset_version(), fLOG=myprint,
            runtime='python', debug=debug))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_DecisionTreeClassifier_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = False
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"DecisionTreeClassifier"},
            opset_min=onnx_opset_version(), fLOG=myprint,
            runtime='python', debug=debug))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_DecisionTreeRegressor_python_64(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 2 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"DecisionTreeRegressor"},
            opset_min=onnx_opset_version(), fLOG=myprint,
            runtime='python', debug=debug, store_models=True,
            filter_exp=lambda m, p: '-64' in p))
        rows = [row for row in rows if 'OK' not in row['available']]
        available = [row['available'] for row in rows]
        if len(available) > 0:
            import pprint
            raise AssertionError(
                "The runtime did have an issue with double\n{}".format(pprint.pformat(rows)))
        self.assertGreater(len(buffer), 1 if debug else 0)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_DecisionTreeRegressor_python_100(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 2 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"DecisionTreeRegressor"},
            opset_min=onnx_opset_version(), fLOG=myprint,
            runtime='python', debug=debug, store_models=True,
            filter_exp=lambda m, p: '-f100' in p))
        self.assertGreater(len(buffer), 1 if debug else 0)
        row = rows[0]
        init = row['init_types'][0][1]
        self.assertEqual(init.shape, (1, 100))


if __name__ == "__main__":
    unittest.main()
