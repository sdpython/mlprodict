"""
@brief      test tree node (time=4s)
"""
import os
import unittest
import pickle
import pandas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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


if __name__ == "__main__":
    unittest.main()
