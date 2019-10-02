"""
A template to benchmark a model
with :epkg:`asv`. The benchmark can be run through
file `run_asv.sh <https://github.com/sdpython/mlprodict/blob/master/run_asv.sh>`_
on Linux or `run_asv.bat
<https://github.com/sdpython/mlprodict/blob/master/run_asv.bat>`_ on
Windows.

.. warning::
    On Windows, you should avoid cloning the repository
    on a folder with a long full name. Visual Studio tends to
    abide by the rule of the maximum path length even though
    the system is told othewise.
"""
import os
from logging import getLogger
import numpy
import pickle
from sklearn import set_config
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt.validate.validate_benchmark import make_n_rows
# from mlprodict.onnrt.validate.validate_problems import _modify_dimension


class TemplateBenchmark:

    params = [
        ['skl', 'pyrt', 'ort'],
        [1, 100, 10000]
    ]
    param_names = ['rt', 'N']

    @property
    def _name(self):
        last = 'cache_{}.pickle'.format(self.__class__.__name__)
        if os.path.exists('_cache'):
            return os.path.join('_cache', last)
        return last

    def setup_cache(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        X = X_test.astype(numpy.float32)
        y = y_test
        model = LogisticRegression(multi_class='ovr', solver='liblinear')
        model.fit(X_train, y_train)
        stored = {'model': model, 'X': X, 'y': y}
        with open(self._name, "wb") as f:
            pickle.dump(stored, f)

    def setup(self, runtime, N):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        if runtime not in TemplateBenchmark.params[0]:
            raise ValueError("Unknown runtime '{}'.".format(runtime))
        set_config(assume_finite=True)
        with open(self._name, "rb") as f:
            stored = pickle.load(f)
        self.stored = stored
        self.model = stored['model']
        self.X, self.y = make_n_rows(stored['X'], N, stored['y'])

        self.onx = to_onnx(self.model, self.X)
        self.oinfpy = OnnxInference(self.onx, runtime='python')
        self.oinfort = OnnxInference(self.onx, runtime='onnxruntime1')

    def time_predict(self, runtime, N):
        if runtime == 'skl':
            return self.model.predict_proba(self.X)
        if runtime == 'ort':
            return self.oinfort.run({'X': self.X})
        if runtime == 'pyrt':
            return self.oinfpy.run({'X': self.X})
        raise ValueError("Unknown runtime '{}'.".format(runtime))

    def peakmem_predict(self, runtime, N):
        if runtime == 'skl':
            return self.model.predict_proba(self.X)
        if runtime == 'ort':
            return self.oinfort.run({'X': self.X})
        if runtime == 'pyrt':
            return self.oinfpy.run({'X': self.X})
        raise ValueError("Unknown runtime '{}'.".format(runtime))

    def track_score(self, runtime, N):
        if runtime == 'skl':
            yp = self.model.predict(self.X)
        elif runtime == 'ort':
            raw = self.oinfort.run({'X': self.X})
            yp = raw['output_label']
        elif runtime == 'pyrt':
            raw = self.oinfpy.run({'X': self.X})
            yp = raw['output_label']
        else:
            raise ValueError("Unknown runtime '{}'.".format(runtime))
        return accuracy_score(self.y, yp)

    def track_onnxsize(self, runtime, N):
        return len(self.onx.SerializeToString())
