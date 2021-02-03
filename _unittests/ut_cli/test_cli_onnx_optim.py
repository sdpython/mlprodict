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
from sklearn.exceptions import ConvergenceWarning
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder, ignore_warnings
from mlprodict.__main__ import main
from mlprodict.cli import convert_validate, onnx_optim


class TestCliOnnxOptim(ExtTestCase):

    def test_cli_onnx_optim(self):
        st = BufferedPrint()
        main(args=["onnx_optim", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("verbose", res)

    @ignore_warnings(ConvergenceWarning)
    def test_onnx_optim(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression()
        clr.fit(X_train, y_train)

        temp = get_temp_folder(__file__, "temp_onnx_optim")
        data = os.path.join(temp, "data.csv")
        pandas.DataFrame(X_test).to_csv(data, index=False)
        pkl = os.path.join(temp, "model.pkl")
        with open(pkl, "wb") as f:
            pickle.dump(clr, f)

        outonnx = os.path.join(temp, 'outolr.onnx')
        convert_validate(pkl=pkl, data=data, verbose=0,
                         method="predict,predict_proba",
                         outonnx=outonnx,
                         name="output_label,output_probability")
        outonnx2 = os.path.join(temp, 'outolr2.onnx')
        st = BufferedPrint()
        onnx_optim(outonnx, outonnx2, verbose=1, fLOG=st.fprint)
        res = str(st)
        self.assertExists(outonnx2)
        self.assertIn('before.size', res)
        self.assertIn('after.size', res)


if __name__ == "__main__":
    unittest.main()
