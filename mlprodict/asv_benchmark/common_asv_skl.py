"""
Common class for all benchmarks testing
converted models from :epkg:`scikit-learn`
with :epkg:`asv`. The benchmark can be run through
file :epkg:`run_asv.sh` on Linux or :epkg:`run_asv.bat` on
Windows.

.. warning::
    On Windows, you should avoid cloning the repository
    on a folder with a long full name. Visual Studio tends to
    abide by the rule of the maximum path length even though
    the system is told otherwise.
"""
import os
from datetime import datetime
import pickle
from logging import getLogger
import numpy
from sklearn import set_config
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score, mean_absolute_error,
    silhouette_score)
from sklearn.model_selection import train_test_split
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import (
    to_onnx, register_rewritten_operators, register_converters)
from mlprodict.onnxrt.validate.validate_benchmark import make_n_rows
from mlprodict.onnxrt.validate.validate_problems import _modify_dimension
from mlprodict.onnx_tools.optim import onnx_statistics
from mlprodict.tools.asv_options_helper import (
    expand_onnx_options, get_opset_number_from_onnx,
    get_ir_version_from_onnx, version2number)
from mlprodict.tools.model_info import set_random_state
from mlprodict.tools.ort_wrapper import onnxrt_version


class _CommonAsvSklBenchmark:
    """
    Common tests to all benchmarks testing converted
    :epkg:`scikit-learn` models. See `benchmark attributes
    <https://asv.readthedocs.io/en/stable/benchmarks.html#general>`_.
    """

    # Part which changes.
    # params and param_names may be changed too.

    params = [
        ['skl', 'pyrtc', 'ort'],  # values for runtime
        [1, 10, 100, 10000],  # values for N
        [4, 20],  # values for nf
        [get_opset_number_from_onnx()],  # values for opset
        ["float", "double"],  # values for dtype
        [None],  # values for optim
    ]
    param_names = ['rt', 'N', 'nf', 'opset', 'dtype', 'optim']
    chk_method_name = None
    version = datetime.now().isoformat()
    pretty_source = "disabled"

    par_ydtype = numpy.int64
    par_dofit = True
    par_convopts = None

    def _create_model(self):  # pragma: no cover
        raise NotImplementedError("This method must be overwritten.")

    def _create_onnx_and_runtime(self, runtime, model, X, opset, dtype, optim):  # pragma: no cover
        raise NotImplementedError("This method must be overwritten.")

    def _score_metric(self, X, y_exp, y_pred):  # pragma: no cover
        raise NotImplementedError("This method must be overwritten.")

    def _optimize_onnx(self, onx):
        return onx

    def _get_xdtype(self, dtype):
        if dtype in ('float', numpy.float32):
            return numpy.float32
        elif dtype in ('double', '64', 64, numpy.float64):
            return numpy.float64
        raise ValueError(  # pragma: no cover
            "Unknown dtype '{}'.".format(dtype))

    def _get_dataset(self, nf, dtype):
        xdtype = self._get_xdtype(dtype)
        data = load_iris()
        X, y = data.data, data.target
        state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
        rnd = state.randn(*X.shape) / 3
        X += rnd
        X = _modify_dimension(X, nf)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        Xt = X_test.astype(xdtype)
        yt = y_test.astype(self.par_ydtype)
        return (X_train, y_train), (Xt, yt)

    def _to_onnx(self, model, X, opset, dtype, optim):
        if optim is None or len(optim) == 0:
            options = self.par_convopts
        elif self.par_convopts and len(self.par_convopts) > 0:
            raise NotImplementedError(  # pragma: no cover
                "Conflict between par_convopts={} and optim={}".format(
                    self.par_convopts, optim))
        else:
            # Expand common onnx options, see _nick_name_options.
            options = expand_onnx_options(model, optim)

        return to_onnx(model, X, options=options, target_opset=opset)

    def _create_onnx_inference(self, onx, runtime):
        if 'onnxruntime' in runtime:
            old = onx.ir_version
            onx.ir_version = get_ir_version_from_onnx()
        else:
            old = None

        try:
            res = OnnxInference(onx, runtime=runtime)
        except RuntimeError as e:  # pragma: no cover
            if "[ONNXRuntimeError]" in str(e):
                return RuntimeError("onnxruntime fails due to {}".format(str(e)))
            raise e
        if old is not None:
            onx.ir_version = old
        return res

    # Part which does not change.

    def _check_rt(self, rt, meth):
        """
        Checks that runtime has the appropriate method.
        """
        if rt is None:
            raise ValueError("rt cannot be empty.")  # pragma: no cover
        if not hasattr(rt, meth):
            raise TypeError(  # pragma: no cover
                "rt of type %r has no method %r." % (type(rt), meth))

    def runtime_name(self, runtime):
        """
        Returns the runtime shortname.
        """
        if runtime == 'skl':
            name = runtime
        elif runtime == 'ort':
            name = 'onnxruntime1'
        elif runtime == 'ort2':
            name = 'onnxruntime2'  # pragma: no cover
        elif runtime == 'pyrt':
            name = 'python'
        elif runtime == 'pyrtc':
            name = 'python_compiled'
        else:
            raise ValueError(  # pragma: no cover
                "Unknown runtime '{}'.".format(runtime))
        return name

    def _name(self, nf, opset, dtype):
        last = 'cache-{}-nf{}-op{}-dt{}.pickle'.format(
            self.__class__.__name__, nf, opset, dtype)
        return last

    def setup_cache(self):
        "asv API"
        for dtype in self.params[4]:
            for opv in self.params[3]:
                for nf in self.params[2]:
                    (X_train, y_train), (X, y) = self._get_dataset(nf, dtype)
                    model = self._create_model()
                    if self.par_dofit:
                        set_random_state(model)
                        model.fit(X_train, y_train)
                    stored = {'model': model, 'X': X, 'y': y}
                    filename = self._name(nf, opv, dtype)
                    with open(filename, "wb") as f:
                        pickle.dump(stored, f)
                    if not os.path.exists(filename):
                        raise RuntimeError(  # pragma: no cover
                            "Unable to dump model %r into %r." % (
                                model, filename))

    def setup(self, runtime, N, nf, opset, dtype, optim):
        "asv API"
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()
        register_rewritten_operators()
        with open(self._name(nf, opset, dtype), "rb") as f:
            stored = pickle.load(f)
        self.stored = stored
        self.model = stored['model']
        self.X, self.y = make_n_rows(stored['X'], N, stored['y'])
        onx, rt_, rt_fct_, rt_fct_track_ = self._create_onnx_and_runtime(
            runtime, self.model, self.X, opset, dtype, optim)
        self.onx = onx
        setattr(self, "rt_" + runtime, rt_)
        setattr(self, "rt_fct_" + runtime, rt_fct_)
        setattr(self, "rt_fct_track_" + runtime, rt_fct_track_)
        set_config(assume_finite=True)

    def time_predict(self, runtime, N, nf, opset, dtype, optim):
        "asv API"
        return getattr(self, "rt_fct_" + runtime)(self.X)

    def peakmem_predict(self, runtime, N, nf, opset, dtype, optim):
        "asv API"
        return getattr(self, "rt_fct_" + runtime)(self.X)

    def track_score(self, runtime, N, nf, opset, dtype, optim):
        "asv API"
        yp = getattr(self, "rt_fct_track_" + runtime)(self.X)
        return self._score_metric(self.X, self.y, yp)

    def track_onnxsize(self, runtime, N, nf, opset, dtype, optim):
        "asv API"
        return len(self.onx.SerializeToString())

    def track_nbnodes(self, runtime, N, nf, opset, dtype, optim):
        "asv API"
        stats = onnx_statistics(self.onx)
        return stats.get('nnodes', 0)

    def track_vmlprodict(self, runtime, N, nf, opset, dtype, optim):
        "asv API"
        from mlprodict import __version__
        return version2number(__version__)

    def track_vsklearn(self, runtime, N, nf, opset, dtype, optim):
        "asv API"
        from sklearn import __version__
        return version2number(__version__)

    def track_vort(self, runtime, N, nf, opset, dtype, optim):
        "asv API"
        return version2number(onnxrt_version)

    def check_method_name(self, method_name):
        "Does some verifications. Fails if inconsistencies."
        if getattr(self, 'chk_method_name', None) not in (None, method_name):
            raise RuntimeError(  # pragma: no cover
                "Method name must be '{}'.".format(method_name))
        if getattr(self, 'chk_method_name', None) is None:
            raise RuntimeError(  # pragma: no cover
                "Unable to check that the method name is correct "
                "(expected is '{}')".format(
                    method_name))


class _CommonAsvSklBenchmarkClassifier(_CommonAsvSklBenchmark):
    """
    Common class for a classifier.
    """
    chk_method_name = 'predict_proba'

    def _score_metric(self, X, y_exp, y_pred):
        return accuracy_score(y_exp, y_pred)

    def _create_onnx_and_runtime(self, runtime, model, X, opset, dtype, optim):
        self.check_method_name('predict_proba')
        onx_ = self._to_onnx(model, X, opset, dtype, optim)
        onx = self._optimize_onnx(onx_)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            rt_fct_ = lambda X: model.predict_proba(X)
            rt_fct_track_ = lambda X: model.predict(X)
        else:
            rt_ = self._create_onnx_inference(onx, name)
            self._check_rt(rt_, 'run')
            rt_fct_ = lambda pX: rt_.run({'X': pX})
            rt_fct_track_ = lambda pX: rt_fct_(pX)['output_label']
        return onx, rt_, rt_fct_, rt_fct_track_


class _CommonAsvSklBenchmarkClassifierRawScore(_CommonAsvSklBenchmark):
    """
    Common class for a classifier.
    """
    chk_method_name = 'decision_function'

    def _score_metric(self, X, y_exp, y_pred):
        return accuracy_score(y_exp, y_pred)

    def _create_onnx_and_runtime(self, runtime, model, X, opset, dtype, optim):
        self.check_method_name('decision_function')
        onx_ = self._to_onnx(model, X, opset, dtype, optim)
        onx = self._optimize_onnx(onx_)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            rt_fct_ = lambda X: model.decision_function(X)
            rt_fct_track_ = lambda X: model.predict(X)
        else:
            rt_ = self._create_onnx_inference(onx, name)
            self._check_rt(rt_, 'run')
            rt_fct_ = lambda X: rt_.run({'X': X})
            rt_fct_track_ = lambda X: rt_fct_(X)['output_label']
        return onx, rt_, rt_fct_, rt_fct_track_


class _CommonAsvSklBenchmarkClustering(_CommonAsvSklBenchmark):
    """
    Common class for a clustering algorithm.
    """
    chk_method_name = 'predict'

    def _score_metric(self, X, y_exp, y_pred):
        if X.shape[0] == 1:
            return 0.  # pragma: no cover
        elif set(y_pred) == 1:
            return 0.  # pragma: no cover
        return silhouette_score(X, y_pred)

    def _create_onnx_and_runtime(self, runtime, model, X, opset, dtype, optim):
        self.check_method_name('predict')
        onx_ = self._to_onnx(model, X, opset, dtype, optim)
        onx = self._optimize_onnx(onx_)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            rt_fct_ = lambda X: model.predict(X.astype(numpy.float64))
            rt_fct_track_ = lambda X: model.predict(X.astype(numpy.float64))
        else:
            rt_ = self._create_onnx_inference(onx, name)
            self._check_rt(rt_, 'run')
            rt_fct_ = lambda X: rt_.run({'X': X})
            rt_fct_track_ = lambda X: rt_fct_(X)['label']
        return onx, rt_, rt_fct_, rt_fct_track_


class _CommonAsvSklBenchmarkMultiClassifier(_CommonAsvSklBenchmark):
    """
    Common class for a multi-classifier.
    """
    chk_method_name = 'predict_proba'

    def _get_dataset(self, nf, dtype):
        xdtype = self._get_xdtype(dtype)
        data = load_iris()
        X, y = data.data, data.target
        state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
        rnd = state.randn(*X.shape) / 3
        X += rnd
        nbclass = len(set(y))
        y_ = numpy.zeros((y.shape[0], nbclass), dtype=y.dtype)
        for i, vy in enumerate(y):
            y_[i, vy] = 1
        y = y_
        X = _modify_dimension(X, nf)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        X = X_test.astype(xdtype)
        y = y_test.astype(self.par_ydtype)
        return (X_train, y_train), (X, y)

    def _score_metric(self, X, y_exp, y_pred):
        return accuracy_score(y_exp.ravel(), y_pred.ravel())

    def _create_onnx_and_runtime(self, runtime, model, X, opset, dtype, optim):
        self.check_method_name('predict_proba')
        onx_ = self._to_onnx(model, X, opset, dtype, optim)
        onx = self._optimize_onnx(onx_)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            rt_fct_ = lambda X: model.predict_proba(X)
            rt_fct_track_ = lambda X: model.predict(X)
        else:
            rt_ = self._create_onnx_inference(onx, name)
            self._check_rt(rt_, 'run')
            rt_fct_ = lambda X: rt_.run({'X': X})
            rt_fct_track_ = lambda X: rt_fct_(X)['output_label']
        return onx, rt_, rt_fct_, rt_fct_track_


class _CommonAsvSklBenchmarkOutlier(_CommonAsvSklBenchmark):
    """
    Common class for outlier detection.
    """
    chk_method_name = 'predict'

    def _score_metric(self, X, y_exp, y_pred):
        return numpy.sum(y_pred) / y_pred.shape[0]

    def _create_onnx_and_runtime(self, runtime, model, X, opset, dtype, optim):
        self.check_method_name('predict')
        onx_ = self._to_onnx(model, X, opset, dtype, optim)
        onx = self._optimize_onnx(onx_)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            rt_fct_ = lambda X: model.predict(X)
            rt_fct_track_ = lambda X: model.predict(X)
        else:
            rt_ = self._create_onnx_inference(onx, name)
            self._check_rt(rt_, 'run')
            rt_fct_ = lambda X: rt_.run({'X': X})
            rt_fct_track_ = lambda X: rt_fct_(X)['scores']
        return onx, rt_, rt_fct_, rt_fct_track_


class _CommonAsvSklBenchmarkRegressor(_CommonAsvSklBenchmark):
    """
    Common class for a regressor.
    """
    chk_method_name = 'predict'

    def _score_metric(self, X, y_exp, y_pred):
        return mean_absolute_error(y_exp, y_pred)

    def _create_onnx_and_runtime(self, runtime, model, X, opset, dtype, optim):
        self.check_method_name('predict')
        onx = self._to_onnx(model, X, opset, dtype, optim)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            rt_fct_ = lambda X: model.predict(X)
            rt_fct_track_ = lambda X: model.predict(X)
        else:
            rt_ = self._create_onnx_inference(onx, name)
            self._check_rt(rt_, 'run')
            rt_fct_ = lambda X: rt_.run({'X': X})
            rt_fct_track_ = lambda X: rt_fct_(X)['variable']
        return onx, rt_, rt_fct_, rt_fct_track_


class _CommonAsvSklBenchmarkTrainableTransform(_CommonAsvSklBenchmark):
    """
    Common class for a trainable transformer.
    """
    chk_method_name = 'transform'

    def _score_metric(self, X, y_exp, y_pred):
        return numpy.sum(y_pred) / y_pred.shape[0]

    def _create_onnx_and_runtime(self, runtime, model, X, opset, dtype, optim):
        self.check_method_name('transform')
        onx_ = self._to_onnx(model, X, opset, dtype, optim)
        onx = self._optimize_onnx(onx_)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            rt_fct_ = lambda X: model.transform(X)
            rt_fct_track_ = lambda X: model.transform(X)
        else:
            rt_ = self._create_onnx_inference(onx, name)
            self._check_rt(rt_, 'run')
            rt_fct_ = lambda X: rt_.run({'X': X})
            rt_fct_track_ = lambda X: rt_fct_(X)['variable']
        return onx, rt_, rt_fct_, rt_fct_track_


class _CommonAsvSklBenchmarkTransform(_CommonAsvSklBenchmark):
    """
    Common class for a transformer.
    """
    chk_method_name = 'transform'

    def _score_metric(self, X, y_exp, y_pred):
        return numpy.sum(y_pred) / y_pred.shape[0]

    def _create_onnx_and_runtime(self, runtime, model, X, opset, dtype, optim):
        self.check_method_name('transform')
        onx_ = self._to_onnx(model, X, opset, dtype, optim)
        onx = self._optimize_onnx(onx_)
        name = self.runtime_name(runtime)
        if name == 'skl':
            rt_ = None
            rt_fct_ = lambda X: model.transform(X)
            rt_fct_track_ = lambda X: model.transform(X)
        else:
            rt_ = self._create_onnx_inference(onx, name)
            self._check_rt(rt_, 'run')
            rt_fct_ = lambda X: rt_.run({'X': X})
            rt_fct_track_ = lambda X: rt_fct_(X)['variable']
        return onx, rt_, rt_fct_, rt_fct_track_


class _CommonAsvSklBenchmarkTransformPositive(_CommonAsvSklBenchmarkTransform):
    """
    Common class for a transformer for positive features.
    """
    chk_method_name = 'transform'

    def _get_dataset(self, nf, dtype):
        xdtype = self._get_xdtype(dtype)
        data = load_iris()
        X, y = data.data, data.target
        state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
        rnd = state.randn(*X.shape) / 3
        X += rnd
        X = _modify_dimension(X, nf)
        X = numpy.abs(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        X = X_test.astype(xdtype)
        y = y_test.astype(self.par_ydtype)
        return (X_train, y_train), (X, y)
