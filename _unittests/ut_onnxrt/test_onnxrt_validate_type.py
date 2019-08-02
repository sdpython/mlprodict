"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import pprint
import numpy
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, ExtTestCase, unittest_require_at_least
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
import skl2onnx
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestOnnxrtValidateType(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def dtype_test_validate_sklearn_operators(self, dtype, models=None):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        verbose = 1 if __name__ == "__main__" else 0

        def myfLOG(*args, **kwargs):
            sargs = str(args)
            if verbose <= 1:
                if "+k" in sargs or "Onnx-" in sargs or "OnnxInference: run" in sargs:
                    return
            fLOG(*args, **kwargs)

        def filter_exp(cl, prob):
            if dtype == numpy.float32:
                return '-64' not in prob and '-cov' not in prob
            if dtype == numpy.float64:
                return '-64' in prob and '-cov' not in prob
            raise NotImplementedError(dtype)

        if models is None:
            models = {
                # 'DecisionTreeRegressor',
                # 'DecisionTreeClassifier',
                'KMeans',
                'LinearRegression',
                'LogisticRegression',
                # 'GaussianProcessRegressor',
            }

        logger = getLogger('skl2onnx')
        logger.disabled = True
        subname = str(dtype).split('.')[-1].strip("'><")
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_" + subname)
        nb = 60
        rows = []
        for _, row in zip(
                range(nb),
                enumerate_validated_operator_opsets(
                    verbose, debug=True, fLOG=myfLOG, dump_folder=temp,
                    models=models, filter_exp=filter_exp,
                    dtype=dtype, opset_min=11, store_models=True)):

            up = {}
            outputs = []
            output = row["ort_outputs"]
            outputs.append(output)
            dtypes = []
            if isinstance(output, numpy.ndarray):
                up['output_type'] = output.dtype
                dtypes.append(output.dtype)
            elif isinstance(output, dict):
                for n, value in output.items():
                    if (isinstance(value, numpy.ndarray) and
                            value.dtype in (numpy.float32, numpy.float64)):
                        up['output_type' + n] = value.dtype
                        dtypes.append(value.dtype)
            row.update(up)
            row["odtypes"] = dtypes

            for dt in dtypes:
                if dt in (dtype, numpy.int32, numpy.int64, numpy.str):
                    continue
                else:
                    raise AssertionError(
                        'Issue with one model {}-{}-{} ({})\n----\n{}\n---\n{}'.format(
                            row['name'], row['problem'], dtypes, dtype,
                            pprint.pformat(row), pprint.pformat(outputs)))
            rows.append(row)

        self.assertGreater(len(rows), 1)

    def test_validate_sklearn_operators_float32(self):
        self.dtype_test_validate_sklearn_operators(numpy.float32)

    def test_validate_sklearn_operators_float64(self):
        self.dtype_test_validate_sklearn_operators(numpy.float64)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_validate_sklearn_operators_float64_gpr(self):
        self.dtype_test_validate_sklearn_operators(
            numpy.float64, {'GaussianProcessRegressor'})


if __name__ == "__main__":
    unittest.main()
