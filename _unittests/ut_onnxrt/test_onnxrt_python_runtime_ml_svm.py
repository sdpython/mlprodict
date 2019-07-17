"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntimeMlSVM(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_openmp_compilation(self):
        from mlprodict.onnxrt.ops_cpu.op_svm_regressor_ import RuntimeSVMRegressor  # pylint: disable=E0611
        ru = RuntimeSVMRegressor()
        r = ru.runtime_options()
        self.assertEqual('OPENMP', r)
        nb = ru.omp_get_max_threads()
        self.assertGreater(nb, 0)

    def test_onnxrt_python_SVR(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SVR()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("SVMRegressor", text)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), ['variable'])
        lexp = clr.predict(X_test)
        self.assertEqual(lexp.shape, y['variable'].shape)
        self.assertEqualArray(lexp, y['variable'], decimal=5)


if __name__ == "__main__":
    unittest.main()
