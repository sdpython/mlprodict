"""
A template to benchmark a model
with :epkg:`asv`.
"""
import numpy
from sklearn import set_config
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx


class TemplateBenchmark:

    def setup(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        self.X, self.y = X_test.astype(numpy.float32), y_test
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        set_config(assume_finite=True)
        self.onx = to_onnx(self.model, X_train.astype(numpy.float32))
        self.oinfpy = OnnxInference(self.onx, runtime='python')
        self.oinfort = OnnxInference(self.onx, runtime='onnxruntime1')

    def time_predict_skl(self):
        return self.model.predict(self.X)

    def time_predict_proba_skl(self):
        return self.model.predict_proba(self.X)

    def time_predict_pyrt(self):
        return self.oinfpy.run({'X': self.X})

    def time_predict_ort(self):
        return self.oinfort.run({'X': self.X})

    def peakmem_predict_skl(self):
        return self.model.predict(self.X)

    def peakmem_predict_proba_skl(self):
        return self.model.predict_proba(self.X)

    def peakmem_predict_pyrt(self):
        return self.oinfpy.run({'X': self.X})

    def peakmem_predict_ort(self):
        return self.oinfort.run({'X': self.X})
