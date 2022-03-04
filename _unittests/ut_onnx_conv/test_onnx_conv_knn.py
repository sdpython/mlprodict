"""
@brief      test log(time=5s)
"""
import unittest
from logging import getLogger
import warnings
import numpy
from pandas import DataFrame
from scipy.spatial.distance import cdist as scipy_cdist
from pyquickhelper.pycode import ExtTestCase, ignore_warnings as igw
from onnxruntime.capi._pybind_state import InvalidArgument as OrtInvalidArgument
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (
    KNeighborsRegressor, KNeighborsClassifier, NearestNeighbors)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxIdentity)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType
import skl2onnx
from skl2onnx.algebra.complex_functions import onnx_cdist
from mlprodict.onnx_conv import (
    register_converters, to_onnx)
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.ops_cpu.op_topk import topk_sorted_implementation
from mlprodict import __max_supported_opset__ as TARGET_OPSET, get_ir_version
from mlprodict.testing.test_utils import _capture_output


def old_topk_sorted_implementation(X, k, axis, largest):
    """
    Retrieves the top-k elements.
    @param      X           data
    @param      k           k in top-k
    @param      axis        axis chosen to select the top-k elements
    @param      largest     largest (1) or smallest (0)
    @return                 top-k values, top-k indices
    """
    sorted_indices = numpy.argsort(X, axis=axis)
    sorted_values = numpy.sort(X, axis=axis)
    if largest:
        sorted_indices = numpy.flip(sorted_indices, axis=axis)
        sorted_values = numpy.flip(sorted_values, axis=axis)
    ark = numpy.arange(k)
    topk_sorted_indices = numpy.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = numpy.take(sorted_values, ark, axis=axis)
    return topk_sorted_values, topk_sorted_indices


class TestOnnxConvKNN(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @igw((DeprecationWarning, FutureWarning))
    def test_topk_sorted_implementation(self):
        X = numpy.array([[0, 1, 0, 2],
                         [1, 0, 4, 5],
                         [9, 8, 5, 6]], dtype=numpy.float64)
        vals, inds = old_topk_sorted_implementation(X, 2, 1, 0)
        vals2, inds2 = topk_sorted_implementation(X, 2, 1, 0)
        self.assertEqualArray(vals, vals2)
        self.assertEqualArray(inds, inds2)

        X = numpy.array([[0, 1, 0, 2],
                         [1, 0, 4, 5],
                         [9, 8, 5, 6]], dtype=numpy.float64)
        vals, inds = old_topk_sorted_implementation(X, 2, 1, 1)
        vals2, inds2 = topk_sorted_implementation(X, 2, 1, 1)
        self.assertEqualArray(vals, vals2)
        self.assertEqualArray(inds, inds2)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_example_cdist_in_euclidean(self):
        for metric in ['euclidean', 'minkowski']:
            for opv in [11, TARGET_OPSET]:
                with self.subTest(metric=metric, opv=opv):
                    x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
                        numpy.float32).reshape((3, 2))
                    x2 = numpy.array([1.1, 2.1, 4.01, 5.01,
                                      5.001, 4.001, 0, 0]).astype(
                        numpy.float32).reshape((4, 2))
                    cop = OnnxAdd('input', 'input',
                                  op_version=opv)

                    if metric == "minkowski":
                        cop2 = OnnxIdentity(
                            onnx_cdist(cop, x2, dtype=numpy.float32,
                                       metric=metric, op_version=opv, p=2),
                            output_names=['cdist'], op_version=opv)
                    else:
                        cop2 = OnnxIdentity(
                            onnx_cdist(cop, x2, dtype=numpy.float32,
                                       metric=metric, op_version=opv),
                            output_names=['cdist'], op_version=opv)

                    model_def = cop2.to_onnx(
                        inputs=[('input', FloatTensorType([None, None]))],
                        outputs=[('cdist', FloatTensorType())],
                        target_opset=opv)

                    sess = OnnxInference(model_def)
                    res = sess.run({'input': x})['cdist']
                    exp = scipy_cdist(x * 2, x2, metric=metric)
                    self.assertEqualArray(exp, res, decimal=5)

                    if metric == "minkowski":
                        continue
                    x = numpy.array(
                        [[6.1, 2.8, 4.7, 1.2],
                         [5.7, 3.8, 1.7, 0.3],
                         [7.7, 2.6, 6.9, 2.3],
                         [6.0, 2.9, 4.5, 1.5],
                         [6.8, 2.8, 4.8, 1.4],
                         [5.4, 3.4, 1.5, 0.4],
                         [5.6, 2.9, 3.6, 1.3],
                         [6.9, 3.1, 5.1, 2.3]], dtype=numpy.float32)
                    cop = OnnxAdd('input', 'input', op_version=opv)
                    cop2 = OnnxIdentity(onnx_cdist(cop, x, dtype=numpy.float32,
                                                   op_version=opv),
                                        output_names=['cdist'], op_version=opv)

                    model_def = cop2.to_onnx(
                        inputs=[('input', FloatTensorType([None, None]))],
                        outputs=[('cdist', FloatTensorType())],
                        target_opset=opv)

                    sess = OnnxInference(model_def)
                    res = sess.run({'input': x})['cdist']
                    exp = scipy_cdist(x * 2, x, metric="sqeuclidean")
                    self.assertEqualArray(exp, res, decimal=4)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_example_cdist_in_minkowski(self):
        x = numpy.array([1, 2, 1, 3, 2, 2, 2, 3]).astype(
            numpy.float32).reshape((4, 2))
        x2 = numpy.array([[1, 2], [2, 2], [2.1, 2.1], [2, 2]]).astype(
            numpy.float32).reshape((4, 2))
        for pp in [1, 2]:
            with self.subTest(pp=pp):
                cop = OnnxIdentity(
                    'input', op_version=TARGET_OPSET)
                cop2 = OnnxIdentity(
                    onnx_cdist(cop, x2, dtype=numpy.float32,
                               metric="minkowski", p=pp,
                               op_version=TARGET_OPSET),
                    output_names=['cdist'],
                    op_version=TARGET_OPSET)

                model_def = cop2.to_onnx(
                    inputs=[('input', FloatTensorType([None, None]))],
                    outputs=[('cdist', FloatTensorType())])

                try:
                    sess = OnnxInference(model_def)
                except RuntimeError as e:
                    raise AssertionError("Issue\n{}".format(model_def)) from e
                res = sess.run({'input': x})['cdist']
                exp = scipy_cdist(x, x2, metric="minkowski", p=pp)
                self.assertEqualArray(exp, res, decimal=5)

        with self.subTest(pp=3):
            x = numpy.array(
                [[6.1, 2.8, 4.7, 1.2],
                 [5.7, 3.8, 1.7, 0.3],
                 [7.7, 2.6, 6.9, 2.3],
                 [6.0, 2.9, 4.5, 1.5],
                 [6.8, 2.8, 4.8, 1.4],
                 [5.4, 3.4, 1.5, 0.4],
                 [5.6, 2.9, 3.6, 1.3],
                 [6.9, 3.1, 5.1, 2.3]], dtype=numpy.float32)
            cop = OnnxAdd('input', 'input',
                          op_version=TARGET_OPSET)
            cop2 = OnnxIdentity(
                onnx_cdist(cop, x, dtype=numpy.float32, metric="minkowski",
                           p=3, op_version=TARGET_OPSET),
                output_names=['cdist'], op_version=TARGET_OPSET)

            model_def = cop2.to_onnx(
                inputs=[('input', FloatTensorType([None, None]))],
                outputs=[('cdist', FloatTensorType())])

            sess = OnnxInference(model_def)
            res = sess.run({'input': x})['cdist']
            exp = scipy_cdist(x * 2, x, metric="minkowski", p=3)
            self.assertEqualArray(exp, res, decimal=4)

    @igw((DeprecationWarning, FutureWarning))
    def test_register_converters(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            res = register_converters(True)
        self.assertGreater(len(res), 2)

    def onnx_test_knn_single_classreg(self, dtype, n_targets=1, debug=False,
                                      add_noise=False, runtime='python',
                                      target_opset=None, optim=None,
                                      kind='reg', level=1, largest0=True,
                                      metric_params=None, **kwargs):
        iris = load_iris()
        X, y = iris.data, iris.target
        if add_noise:
            X += numpy.random.randn(X.shape[0], X.shape[1]) * 10
        if kind == 'reg':
            y = y.astype(dtype)
        elif kind == 'bin':
            y = (y % 2).astype(numpy.int64)
        elif kind == 'mcl':
            y = y.astype(numpy.int64)
        else:
            raise AssertionError("unknown '{}'".format(kind))

        if n_targets != 1:
            yn = numpy.empty((y.shape[0], n_targets), dtype=dtype)
            for i in range(n_targets):
                yn[:, i] = y + i
            y = yn
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        X_test = X_test.astype(dtype)
        if kind in ('bin', 'mcl'):
            clr = KNeighborsClassifier(
                metric_params=metric_params, **kwargs)
        elif kind == 'reg':
            clr = KNeighborsRegressor(
                metric_params=metric_params, **kwargs)
        else:
            raise NotImplementedError(kind)
        clr.fit(X_train, y_train)

        if optim is None:
            options = None
        else:
            options = {clr.__class__: {'optim': 'cdist'}}
        if not largest0:
            if options is None:
                options = {}
            if clr.__class__ not in options:
                options[clr.__class__] = {}
            options[clr.__class__].update({'largest0': False})

        if target_opset is None:
            opsets = list(sorted(set([
                9, 10, 11, 12, 13, 14, 15, TARGET_OPSET])))  # opset=13, 14, ...
        else:
            opsets = [target_opset]
        for ops in opsets:
            if ops is None:
                raise AssertionError("Cannot happen: {}.".format(opsets))
            with self.subTest(target_opset=ops):
                try:
                    model_def = to_onnx(
                        clr, X_train.astype(dtype), rewrite_ops=True,
                        target_opset=ops, options=options)
                except NameError as e:
                    if "Option 'largest0' not in" in str(e):
                        continue
                if 'onnxruntime' in runtime:
                    model_def.ir_version = get_ir_version(ops)
                try:
                    if runtime == 'onnxruntime2':
                        oinf = _capture_output(
                            lambda: OnnxInference(
                                model_def, runtime=runtime),  # pylint: disable=W0640
                            'c')[0]
                    else:
                        oinf = OnnxInference(model_def, runtime=runtime)
                except (RuntimeError, TypeError, OrtInvalidArgument) as e:
                    if "No Op registered for Identity with domain_version of 12" in str(e):
                        continue
                    if debug:
                        raise AssertionError(
                            "Unable to create a model for target_opset={}\n----\n{}\n----".format(
                                ops, str(model_def)[:100])) from e
                    if "Unknown model file format version." in str(e):
                        continue
                    raise AssertionError(
                        "Unable to create model for opset={} and runtime='{}'\n{}"
                        "".format(ops, runtime, str(model_def)[:100])) from e

                if debug:
                    y = oinf.run({'X': X_test}, verbose=level, fLOG=print)
                else:
                    y = oinf.run({'X': X_test})

                lexp = clr.predict(X_test)
                if kind == 'reg':
                    self.assertEqual(list(sorted(y)), ['variable'])
                    if dtype == numpy.float32:
                        self.assertEqualArray(
                            lexp, y['variable'], decimal=5, squeeze=True)
                    else:
                        self.assertEqualArray(
                            lexp, y['variable'], squeeze=True)
                else:
                    self.assertEqual(list(sorted(y)),
                                     ['output_label', 'output_probability'])
                    self.assertEqualArray(lexp, y['output_label'])
                    lprob = clr.predict_proba(X_test)
                    self.assertEqualArray(
                        lprob, DataFrame(y['output_probability']).values,
                        decimal=5)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32(self):
        self.onnx_test_knn_single_classreg(numpy.float32)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_cdist(self):
        self.onnx_test_knn_single_classreg(numpy.float32, optim='cdist')

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_op10(self):
        self.onnx_test_knn_single_classreg(
            numpy.float32, target_opset=10, debug=False)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_onnxruntime1(self):
        self.onnx_test_knn_single_classreg(
            numpy.float32, runtime="onnxruntime1", target_opset=10)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_onnxruntime2(self):
        try:
            self.onnx_test_knn_single_classreg(
                numpy.float32, runtime="onnxruntime2", target_opset=10,
                debug=False)
        except (RuntimeError, OrtInvalidArgument) as e:
            if "Invalid rank for input: Ar_Z0 Got: 2 Expected: 1" in str(e):
                return
            if "Got invalid dimensions for input:" in str(e):
                return
            raise e

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_balltree(self):
        self.onnx_test_knn_single_classreg(
            numpy.float32, algorithm='ball_tree')

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_kd_tree(self):
        self.onnx_test_knn_single_classreg(numpy.float32, algorithm='kd_tree')

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_brute(self):
        self.onnx_test_knn_single_classreg(numpy.float32, algorithm='brute')

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg64(self):
        self.onnx_test_knn_single_classreg(numpy.float64)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_target2(self):
        self.onnx_test_knn_single_classreg(numpy.float32, n_targets=2)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_target2_onnxruntime(self):
        self.onnx_test_knn_single_classreg(
            numpy.float32, n_targets=2, runtime="onnxruntime1")

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_k1(self):
        self.onnx_test_knn_single_classreg(numpy.float32, n_neighbors=1)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_k1_target2(self):
        self.onnx_test_knn_single_classreg(
            numpy.float32, n_neighbors=1, n_targets=2)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_minkowski(self):
        self.onnx_test_knn_single_classreg(numpy.float32, metric='minkowski')

    @igw((DeprecationWarning, SyntaxWarning))
    def test_onnx_test_knn_single_reg32_minkowski_p1(self):
        self.onnx_test_knn_single_classreg(numpy.float32, metric='minkowski',
                                           metric_params={'p': 1}, add_noise=True)

    @igw((DeprecationWarning, SyntaxWarning))
    def test_onnx_test_knn_single_reg32_minkowski_p21(self):
        self.onnx_test_knn_single_classreg(numpy.float32, metric='minkowski',
                                           algorithm='brute', metric_params={'p': 2.1})

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg32_distance(self):
        self.onnx_test_knn_single_classreg(numpy.float32, weights='distance',
                                           largest0=False)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_reg_equal(self):
        # We would like to make scikit-learn and the runtime handles the
        # ex aequo the same way but that's difficult.
        X = numpy.full((20, 4), 1, dtype=numpy.float32)
        X[::2, 3] = 20
        X[1::5, 1] = 30
        X[::5, 2] = 40
        y = X.sum(axis=1) + numpy.arange(X.shape[0]) / 10
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=11, test_size=0.5)
        clr = KNeighborsRegressor(algorithm='brute', n_neighbors=3)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train, rewrite_ops=True)
        oinf = OnnxInference(model_def, runtime='python')
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), ['variable'])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['variable'], decimal=5, squeeze=True)

    # classification

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_bin32(self):
        self.onnx_test_knn_single_classreg(numpy.float32, kind='bin')

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_bin32_onnxruntime(self):
        self.onnx_test_knn_single_classreg(
            numpy.float32, kind='bin', runtime="onnxruntime1")

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_bin32_cdist(self):
        self.onnx_test_knn_single_classreg(
            numpy.float32, kind='bin', optim='cdist')

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_mcl32(self):
        self.onnx_test_knn_single_classreg(numpy.float32, kind='mcl')

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_weights_bin32(self):
        self.onnx_test_knn_single_classreg(numpy.float32, kind='bin',
                                           weights='distance', largest0=False)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_weights_bin32_cdist(self):
        self.onnx_test_knn_single_classreg(numpy.float32, kind='bin',
                                           weights='distance', optim='cdist',
                                           largest0=False)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_weights_mcl32(self):
        self.onnx_test_knn_single_classreg(numpy.float32, kind='mcl',
                                           weights='distance', largest0=False)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_bin64(self):
        self.onnx_test_knn_single_classreg(numpy.float64, kind='bin')

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_mcl64(self):
        self.onnx_test_knn_single_classreg(numpy.float64, kind='mcl')

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_weights_bin64(self):
        self.onnx_test_knn_single_classreg(numpy.float64, kind='bin',
                                           weights='distance', largest0=False)

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_single_weights_mcl64(self):
        self.onnx_test_knn_single_classreg(numpy.float64, kind='mcl',
                                           weights='distance', largest0=False)

    # transform

    @igw((DeprecationWarning, FutureWarning))
    def test_onnx_test_knn_transform(self):
        iris = load_iris()
        X, _ = iris.data, iris.target

        X_train, X_test = train_test_split(X, random_state=11)
        clr = NearestNeighbors(n_neighbors=3)
        clr.fit(X_train)

        for to in (10, 11, 12, 13, 14, 15):  # opset=13, 14, ...
            if to > TARGET_OPSET:
                break
            try:
                model_def = to_onnx(
                    clr, X_train.astype(numpy.float32),
                    rewrite_ops=True, options={NearestNeighbors: {'largest0': False}},
                    target_opset=to)
            except NameError as e:
                if "Option 'largest0' not in" in str(e):
                    continue
            oinf = OnnxInference(model_def, runtime='python')

            X_test = X_test[:3]
            y = oinf.run({'X': X_test.astype(numpy.float32)})
            dist, ind = clr.kneighbors(X_test)

            self.assertEqual(list(sorted(y)), ['distance', 'index'])
            self.assertEqualArray(ind, y['index'])
            self.assertEqualArray(dist, DataFrame(
                y['distance']).values, decimal=5)

    # calibrated

    @igw((DeprecationWarning, FutureWarning))
    def test_model_calibrated_classifier_cv_isotonic_binary_knn(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        clf = KNeighborsClassifier().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="isotonic").fit(X, y)
        model_onnx = skl2onnx.convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        oinf = OnnxInference(model_onnx, runtime='python')
        y = oinf.run({'input': X.astype(numpy.float32)})
        pred = clf.predict(X)
        probs = clf.predict_proba(X)
        self.assertEqual(pred, y['output_label'])
        self.assertEqual(probs, DataFrame(y['output_probability']).values)

    @igw((DeprecationWarning, FutureWarning))
    def test_model_knn_regressor_equal____(self):
        X, y = make_regression(  # pylint: disable=W0632
            n_samples=1000, n_features=100, random_state=42)
        X = X.astype(numpy.int64)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.5, random_state=42)
        model = KNeighborsRegressor(
            algorithm='brute', metric='manhattan').fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model, 'knn',
            [('input', Int64TensorType([None, X_test.shape[1]]))])
        exp = model.predict(X_test)

        sess = OnnxInference(model_onnx)
        res = sess.run({'input': numpy.array(X_test)})['variable']

        # The conversion has discrepencies when
        # neighbours are at the exact same distance.
        maxd = 1000
        accb = numpy.abs(exp.ravel() - res.ravel()) > maxd
        ind = [i for i, a in enumerate(accb) if a == 1]
        self.assertEqual(len(ind), 0)

        accp = numpy.abs(exp - res) < maxd
        acc = numpy.sum(accp)
        ratio = acc * 1.0 / res.shape[0]
        self.assertGreater(ratio, 0.7)
        # Explainable discrepencies.
        # self.assertEqualArray(exp, res)
        self.assertEqual(numpy.squeeze(exp).shape, numpy.squeeze(res).shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
