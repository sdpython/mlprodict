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
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.__main__ import main
from mlprodict.cli import convert_validate


class TestCliOnnxStats(ExtTestCase):

    def test_cli_onnx_stats(self):
        st = BufferedPrint()
        main(args=["onnx_stats", "--help"], fLOG=st.fprint)
        res = str(st)
        self.assertIn("optim", res)

    def test_onnx_stats(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression()
        clr.fit(X_train, y_train)

        temp = get_temp_folder(__file__, "temp_onnx_stats")
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
        st = BufferedPrint()
        main(args=["onnx_stats", "--name", outonnx],
             fLOG=st.fprint)
        res = str(st)
        self.assertIn("ninits: 0", res)


if __name__ == "__main__":
    unittest.main()
