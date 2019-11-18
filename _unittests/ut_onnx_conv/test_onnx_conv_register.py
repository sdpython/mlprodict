"""
@brief      test log(time=2s)
"""
import unittest
import warnings
from pyquickhelper.pycode import ExtTestCase
from xgboost import XGBRegressor, XGBClassifier  # pylint: disable=C0411
from mlprodict.onnx_conv import register_converters
from mlprodict.onnx_conv.validate_scenarios import find_suitable_problem
from mlprodict.onnxrt.validate.validate import find_suitable_problem as main_find_suitable_problem
from mlprodict.onnxrt import sklearn_operators


class TestOnnxConvRegister(ExtTestCase):

    def test_find_suitable_problem(self):
        found = find_suitable_problem(XGBRegressor)
        self.assertEqual(found, ['b-reg', '~b-reg-64'])
        found = find_suitable_problem(XGBClassifier)
        self.assertEqual(found, ['b-cl', 'm-cl', '~b-cl-64'])

    def test_register_converters(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            res = register_converters(True)
        self.assertGreater(len(res), 2)

    def test_register_converters_skl_op(self):
        res = sklearn_operators(extended=True)
        names = set(_['name'] for _ in res)
        self.assertIn('LGBMClassifier', names)
        self.assertIn('LGBMRegressor', names)
        self.assertIn('XGBClassifier', names)
        self.assertIn('XGBRegressor', names)

    def test_sklearn_operators(self):
        res = sklearn_operators(extended=True)
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 4)

    def test_sklearn_operator_here(self):
        subfolders = ['ensemble'] + ['mlprodict.onnx_conv']
        for sub in sorted(subfolders):
            models = sklearn_operators(sub)
            if len(models) == 0:
                raise AssertionError(
                    "models is empty for subfolder '{}'.".format(sub))
            if sub == "mlprodict.onnx_conv":
                names = set(_['name'] for _ in models)
                self.assertIn("LGBMClassifier", names)

    def test_check_whole_model_list(self):
        res = sklearn_operators(extended=True)
        rows = []
        for model in res:
            name = model['name']
            row = dict(name=name)
            try:
                prob = main_find_suitable_problem(model['cl'])
                row['prob'] = prob
            except RuntimeError:
                pass
            rows.append(row)
        set_names = set(_['name'] for _ in rows)
        names = list(_['name'] for _ in rows)
        self.assertEqual(len(set_names), len(names))
        xgb_reg = [_ for _ in rows if _['name'] == 'XGBRegressor']
        self.assertEqual(len(xgb_reg), 1)
        xgb_reg = xgb_reg[0]
        exp = find_suitable_problem(XGBRegressor)
        self.assertEqual(list(sorted(exp)), list(sorted(xgb_reg['prob'])))


if __name__ == "__main__":
    unittest.main()
