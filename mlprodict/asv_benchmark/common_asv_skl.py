"""
Common class for all benchmarks testing
converted models from :epkg:`scikit-learn`
with :epkg:`asv`. The benchmark can be run through
file `run_asv.sh <https://github.com/sdpython/mlprodict/blob/master/run_asv.sh>`_
on Linux or `run_asv.bat
<https://github.com/sdpython/mlprodict/blob/master/run_asv.bat>`_ on
Windows.

.. warning::
    On Windows, you should avoid cloning the repository
    on a folder with a long full name. Visual Studio tends to
    abide by the rule of the maximum path length even though
    the system is told otherwise.
"""
import os
from logging import getLogger
import numpy
import pickle
from sklearn import set_config
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt.validate.validate_benchmark import make_n_rows
from mlprodict.onnxrt.validate.validate_problems import _modify_dimension
from mlprodict.onnxrt.optim import onnx_statistics


class _CommonAsvSklBenchmark:
    """
    Common tests to all benchmarks testing converted
    :epkg:`scikit-learn` models.
    """

    # Part which changes.
    # params and param_names may be changed too.

    params = [
        ['skl', 'pyrt', 'ort'],  # values for runtime
        [1, 100, 10000],  # values for N
        [4, 20],  # values for nf
    ]
    param_names = ['rt', 'N', 'nf']

    xtest_dtype = numpy.float32
    ytest_dtype = numpy.int64

    def _create_model(self):
        raise NotImplementedError("This method must be overwritten.")

    def _create_onnx_and_runtime(self, runtime, model, X):
        raise NotImplementedError("This method must be overwritten.")

    def _score_metric(self, y_exp, y_pred):
        raise NotImplementedError("This method must be overwritten.")

    def _optimize_onnx(self, onx):
        return onx

    def _get_dataset(self, nf):
        data = load_iris()
        X, y = data.data, data.target
        X = _modify_dimension(X, nf)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        X = X_test.astype(self.xtest_dtype)
        y = y_test.astype(self.ytest_dtype)
        return (X_train, y_train), (X, y)

    # Part which does not change.

    def runtime_name(self, runtime):
        if runtime == 'skl':
            name = runtime
        elif runtime == 'ort':
            name = 'onnxruntime1'
        elif runtime == 'ort2':
            name = 'onnxruntime2'
        elif runtime == 'pyrt':
            name = 'python'
        else:
            raise ValueError("Unknown runtime '{}'.".format(runtime))
        return name

    def _name(self, nf):
        last = 'cache-{}-{}.pickle'.format(self.__class__.__name__, nf)
        if os.path.exists('_cache'):
            return os.path.join('_cache', last)
        return last

    def setup_cache(self):
        for nf in self.params[2]:
            (X_train, y_train), (X, y) = self._get_dataset(nf)
            model = self._create_model()
            model.fit(X_train, y_train)
            stored = {'model': model, 'X': X, 'y': y}
            with open(self._name(nf), "wb") as f:
                pickle.dump(stored, f)

    def setup(self, runtime, N, nf):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        set_config(assume_finite=True)
        with open(self._name(nf), "rb") as f:
            stored = pickle.load(f)
        self.stored = stored
        self.model = stored['model']
        self.X, self.y = make_n_rows(stored['X'], N, stored['y'])
        onx, rt_, rt_fct_, rt_fct_track_ = self._create_onnx_and_runtime(
            runtime, self.model, self.X)
        self.onx = onx
        setattr(self, "rt_" + runtime, rt_)
        setattr(self, "rt_fct_" + runtime, rt_fct_)
        setattr(self, "rt_fct_track_" + runtime, rt_fct_track_)

    def time_predict(self, runtime, N, nf):
        return getattr(self, "rt_fct_" + runtime)(self.X)

    def peakmem_predict(self, runtime, N, nf):
        return getattr(self, "rt_fct_" + runtime)(self.X)

    def track_score(self, runtime, N, nf):
        yp = getattr(self, "rt_fct_track_" + runtime)(self.X)
        return self._score_metric(self.y, yp)

    def track_onnxsize(self, runtime, N, nf):
        return len(self.onx.SerializeToString())

    def track_nbnodes(self, runtime, N, nf):
        stats = onnx_statistics(self.onx)
        return stats.get('nnodes', 0)

    def track_opset(self, runtime, N, nf):
        stats = onnx_statistics(self.onx)
        return stats.get('', 0)


class _CommonAsvSklBenchmarkClassifier(_CommonAsvSklBenchmark):
    """
    Common function for a classifier.
    """

    def _score_metric(self, y_exp, y_pred):
        return accuracy_score(y_exp, y_pred)

    def _create_onnx_and_runtime(self, runtime, model, X):
        onx = to_onnx(model, X)
        onx = self._optimize_onnx(onx)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            def rt_fct_(X): return model.predict_proba(X)
            def rt_fct_track_(X): return model.predict(X)
        else:
            rt_ = OnnxInference(onx, runtime=name)
            def rt_fct_(X): return rt_.run({'X': X})
            def rt_fct_track_(X): return rt_fct_(X)['output_label']
        return onx, rt_, rt_fct_, rt_fct_track_


class _CommonAsvSklBenchmarkRegressor(_CommonAsvSklBenchmark):
    """
    Common function for a regressor.
    """

    def _score_metric(self, y_exp, y_pred):
        return r2_score(y_exp, y_pred)

    def _create_onnx_and_runtime(self, runtime, model, X):
        onx = to_onnx(model, X)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            def rt_fct_(X): return model.predict(X)
            def rt_fct_track_(X): return model.predict(X)
        else:
            rt_ = OnnxInference(onx, runtime=name)
            def rt_fct_(X): return rt_.run({'X': X})
            def rt_fct_track_(X): return rt_fct_(X)['variable']
        return onx, rt_, rt_fct_, rt_fct_track_
