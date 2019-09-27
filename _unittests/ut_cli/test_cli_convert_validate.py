"""
@brief      test tree node (time=4s)
"""
import os
import unittest
import pickle
import pandas
import onnx
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.testing import ignore_warnings
import skl2onnx
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder, unittest_require_at_least
from mlprodict.__main__ import main
from mlprodict.cli import convert_validate


class TestCliConvertValidate(ExtTestCase):

    def test_cli_convert_validate(self):
        st = BufferedPrint()
        main(args=["convert_validate", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("verbose", res)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @ignore_warnings(category=(UserWarning, ))
    def test_convert_validate(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression()
        clr.fit(X_train, y_train)

        temp = get_temp_folder(__file__, "temp_convert_validate")
        data = os.path.join(temp, "data.csv")
        pandas.DataFrame(X_test).to_csv(data, index=False)
        pkl = os.path.join(temp, "model.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(clr, f)

        res = convert_validate(pkl=pkl, data=data, verbose=0,
                               method="predict,predict_proba",
                               name="output_label,output_probability")
        self.assertIsInstance(res, dict)
        self.assertLess(res['metrics'][0], 1e-5)
        self.assertLess(res['metrics'][1], 1e-5)

        self.assertRaise(lambda: convert_validate(pkl=pkl, data=data, verbose=0,
                                                  method="predict,predict_proba",
                                                  name="output_label"), ValueError)
        self.assertRaise(lambda: convert_validate(pkl=pkl, data=data, verbose=0,
                                                  method="predict,predict_probas",
                                                  name="output_label,output_probability"), AttributeError)
        self.assertRaise(lambda: convert_validate(pkl=pkl, data=data, verbose=0,
                                                  method="predict,predict_proba",
                                                  name="output_label,output_probabilities"), KeyError)

        res = convert_validate(pkl=pkl, data=data, verbose=0,
                               method="predict,predict_proba",
                               name="output_label,output_probability",
                               noshape=True)
        self.assertIsInstance(res, dict)
        self.assertLess(res['metrics'][0], 1e-5)
        self.assertLess(res['metrics'][1], 1e-5)

    @ignore_warnings(category=(UserWarning, ))
    def test_cli_convert_validater_run(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression()
        clr.fit(X_train, y_train)

        temp = get_temp_folder(__file__, "temp_cli_convert_validate_run")
        data = os.path.join(temp, "data.csv")
        pandas.DataFrame(X_test).to_csv(data, index=False)
        pkl = os.path.join(temp, "model.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(clr, f)

        res = convert_validate(pkl=pkl, data=data, verbose=0,
                               method="predict,predict_proba",
                               name="output_label,output_probability")
        st = BufferedPrint()
        args = ["convert_validate", "--pkl", pkl, '--data', data,
                '--method', "predict,predict_proba",
                '--name', "output_label,output_probability",
                '--verbose', '1']
        main(args, fLOG=st.fprint)
        res = str(st)
        self.assertIn(
            "[convert_validate] compute predictions with method 'predict_proba'", res)

    @ignore_warnings(category=(UserWarning, ))
    def test_cli_convert_validater_switch(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression()
        clr.fit(X_train, y_train)

        temp = get_temp_folder(__file__, "temp_cli_convert_validate_switch")
        data = os.path.join(temp, "data.csv")
        pandas.DataFrame(X_test).to_csv(data, index=False)
        pkl = os.path.join(temp, "model.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(clr, f)

        res = convert_validate(pkl=pkl, data=data, verbose=0,
                               method="predict,predict_proba",
                               name="output_label,output_probability")
        st = BufferedPrint()
        args = ["convert_validate", "--pkl", pkl, '--data', data,
                '--method', "predict,predict_proba",
                '--name', "output_label,output_probability",
                '--verbose', '1', '--use_double', 'switch']
        main(args, fLOG=st.fprint)
        res = str(st)
        self.assertIn(
            "[convert_validate] compute predictions with method 'predict_proba'", res)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @ignore_warnings(category=(UserWarning, ))
    def test_cli_convert_validater_float64(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression()
        clr.fit(X_train, y_train)

        temp = get_temp_folder(__file__, "temp_cli_convert_validate_float64")
        data = os.path.join(temp, "data.csv")
        pandas.DataFrame(X_test).to_csv(data, index=False)
        pkl = os.path.join(temp, "model.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(clr, f)

        res = convert_validate(pkl=pkl, data=data, verbose=0,
                               method="predict,predict_proba",
                               name="output_label,output_probability")
        st = BufferedPrint()
        args = ["convert_validate", "--pkl", pkl, '--data', data,
                '--method', "predict,predict_proba",
                '--name', "output_label,output_probability",
                '--verbose', '1', '--use_double', 'float64']
        main(args, fLOG=st.fprint)
        res = str(st)
        self.assertIn(
            "[convert_validate] compute predictions with method 'predict_proba'", res)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @ignore_warnings(category=(UserWarning, ))
    def test_cli_convert_validater_float64_gpr(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = GaussianProcessRegressor()
        clr.fit(X_train, y_train)

        temp = get_temp_folder(
            __file__, "temp_cli_convert_validate_float64_gpr")
        monx = os.path.join(temp, "gpr.onnx")
        data = os.path.join(temp, "data.csv")
        pandas.DataFrame(X_test).to_csv(data, index=False)
        pkl = os.path.join(temp, "model.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(clr, f)

        try:
            res = convert_validate(
                pkl=pkl, data=data, verbose=0,
                method="predict", name="GPmean", use_double='float64',
                options="{GaussianProcessRegressor:{'optim':'cdist'}}")
        except RuntimeError as e:
            if "requested version 10 < 11 schema version" in str(e):
                return
            raise e
        self.assertNotEmpty(res)
        st = BufferedPrint()
        args = ["convert_validate", "--pkl", pkl, '--data', data,
                '--method', "predict", '--name', "GPmean",
                '--verbose', '1', '--use_double', 'float64',
                '--options', "{GaussianProcessRegressor:{'optim':'cdist'}}",
                '--outonnx', monx]
        main(args, fLOG=st.fprint)
        res = str(st)
        self.assertExists(monx)
        with open(monx, 'rb') as f:
            model = onnx.load(f)
        self.assertIn('CDist', str(model))

    @ignore_warnings(category=(UserWarning, ))
    def test_cli_convert_validater_nodata(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression()
        clr.fit(X_train, y_train)

        temp = get_temp_folder(__file__, "temp_cli_convert_validate_nodata")
        data = os.path.join(temp, "data.csv")
        pandas.DataFrame(X_test).to_csv(data, index=False)
        pkl = os.path.join(temp, "model.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(clr, f)

        res = convert_validate(pkl=pkl, data=None, verbose=0,
                               method="predict,predict_proba",
                               name="output_label,output_probability")
        st = BufferedPrint()
        args = ["convert_validate", "--pkl", pkl,
                '--method', "predict,predict_proba",
                '--name', "output_label,output_probability",
                '--verbose', '1']
        main(args, fLOG=st.fprint)
        res = str(st)
        self.assertNotIn("[convert_validate] compute predictions", res)

    @ignore_warnings(category=(UserWarning, ))
    def test_cli_convert_validater_pkl_nodata(self):
        temp = get_temp_folder(
            __file__, "temp_cli_convert_validate_pkl_nodata")
        monx = os.path.join(temp, "gpr.onnx")
        pkl = os.path.join(temp, "booster.pickle")
        if not os.path.exists(pkl):
            return

        st = BufferedPrint()
        res = convert_validate(pkl=pkl, data=None, verbose=0,
                               method="predict,predict_proba",
                               name="output_label,output_probability",
                               outonnx=monx, fLOG=st.fprint)
        res = str(st)
        self.assertNotIn("[convert_validate] compute predictions", res)


if __name__ == "__main__":
    TestCliConvertValidate().test_cli_convert_validater_pkl_nodata()
    unittest.main()
