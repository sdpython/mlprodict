"""
@brief      test log(time=9s)
"""

import unittest
import numpy
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import (
    LinearRegression, BayesianRidge, SGDRegressor, TheilSenRegressor,
    ARDRegression, ElasticNet, HuberRegressor, Lars, LarsCV, LassoCV, Ridge,
    RANSACRegressor, LassoLars, OrthogonalMatchingPursuitCV,
    LassoLarsIC, LassoLarsCV, PassiveAggressiveRegressor,
    ElasticNetCV, MultiTaskLassoCV, MultiTaskLasso,
    MultiTaskElasticNet, OrthogonalMatchingPursuit,
    MultiTaskElasticNetCV)
from sklearn.svm import LinearSVR
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType, DoubleTensorType, BooleanTensorType)
from mlprodict.onnxrt import OnnxInference
from mlprodict.testing.test_utils import (
    dump_data_and_model, fit_regression_model)


class TestGLMRegressorConverter(ExtTestCase):

    def test_model_linear_regression(self):
        model, X = fit_regression_model(LinearRegression())
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegression-Dec4")

    def test_model_linear_regression_multiple(self):
        model, X = fit_regression_model(LinearRegression(), n_targets=2)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegressionM-Dec4")

    def test_model_linear_regression64(self):
        model, X = fit_regression_model(LinearRegression())
        model_onnx = convert_sklearn(model, "linear regression",
                                     [("input", DoubleTensorType(X.shape))])
        self.assertIsNotNone(model_onnx)
        self.assertIn("elem_type: 11", str(model_onnx))
        dump_data_and_model(
            X.astype(numpy.float64), model, model_onnx,
            basename="SklearnLinearRegression64-Dec4")

    def test_model_linear_regression64_multiple(self):
        model, X = fit_regression_model(LinearRegression(), n_targets=2)
        model_onnx = convert_sklearn(model, "linear regression",
                                     [("input", DoubleTensorType(X.shape))])
        self.assertIsNotNone(model_onnx)
        self.assertIn("elem_type: 11", str(model_onnx))
        dump_data_and_model(
            X.astype(numpy.float64), model, model_onnx,
            basename="SklearnLinearRegression64M-Dec4")

    def test_model_linear_regression_int(self):
        model, X = fit_regression_model(
            LinearRegression(), is_int=True)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegressionInt-Dec4")

    def test_model_linear_regression_nointercept(self):
        model, X = fit_regression_model(
            LinearRegression(fit_intercept=False))
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegressionNoIntercept-Dec4")

    def test_model_linear_regression_bool(self):
        model, X = fit_regression_model(
            LinearRegression(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", BooleanTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegressionBool")

    def test_model_linear_svr(self):
        model, X = fit_regression_model(LinearSVR())
        model_onnx = convert_sklearn(
            model, "linear SVR",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSvr-Dec4")

    def test_model_linear_svr_int(self):
        model, X = fit_regression_model(LinearSVR(), is_int=True)
        model_onnx = convert_sklearn(
            model, "linear SVR",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSvrInt-Dec4")

    def test_model_linear_svr_bool(self):
        model, X = fit_regression_model(LinearSVR(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "linear SVR",
            [("input", BooleanTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSVRBool")

    def test_model_ridge(self):
        model, X = fit_regression_model(Ridge())
        model_onnx = convert_sklearn(
            model, "ridge regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidge-Dec4")

    def test_model_ridge_int(self):
        model, X = fit_regression_model(Ridge(), is_int=True)
        model_onnx = convert_sklearn(
            model, "ridge regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeInt-Dec4")

    def test_model_ridge_bool(self):
        model, X = fit_regression_model(Ridge(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "ridge regression",
            [("input", BooleanTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeBool")

    def test_model_sgd_regressor(self):
        model, X = fit_regression_model(SGDRegressor())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSGDRegressor-Dec4")

    def test_model_sgd_regressor_int(self):
        model, X = fit_regression_model(
            SGDRegressor(), is_int=True)
        model_onnx = convert_sklearn(
            model, "SGD regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSGDRegressorInt-Dec4")

    def test_model_sgd_regressor_bool(self):
        model, X = fit_regression_model(
            SGDRegressor(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "SGD regression",
            [("input", BooleanTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSGDRegressorBool-Dec4")

    def test_model_elastic_net_regressor(self):
        model, X = fit_regression_model(ElasticNet())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnElasticNet-Dec4")

    def test_model_elastic_net_cv_regressor(self):
        model, X = fit_regression_model(ElasticNetCV())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnElasticNetCV-Dec4")

    def test_model_elastic_net_regressor_int(self):
        model, X = fit_regression_model(ElasticNet(), is_int=True)
        model_onnx = convert_sklearn(
            model, "elastic net regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnElasticNetRegressorInt-Dec4")

    def test_model_elastic_net_regressor_bool(self):
        model, X = fit_regression_model(
            ElasticNet(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "elastic net regression",
            [("input", BooleanTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnElasticNetRegressorBool")

    def test_model_lars(self):
        model, X = fit_regression_model(Lars())
        model_onnx = convert_sklearn(
            model, "lars",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLars-Dec4")

    def test_model_lars_cv(self):
        model, X = fit_regression_model(LarsCV())
        model_onnx = convert_sklearn(
            model, "lars",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLarsCV-Dec4")

    def test_model_lasso_lars(self):
        model, X = fit_regression_model(LassoLars(alpha=0.01))
        model_onnx = convert_sklearn(
            model, "lasso lars",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLars-Dec4")

    def test_model_lasso_lars_cv(self):
        model, X = fit_regression_model(LassoLarsCV())
        model_onnx = convert_sklearn(
            model, "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLarsCV-Dec4")

    def test_model_lasso_lars_ic(self):
        model, X = fit_regression_model(LassoLarsIC())
        model_onnx = convert_sklearn(
            model, "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLarsIC-Dec4")

    def test_model_lasso_cv(self):
        model, X = fit_regression_model(LassoCV())
        model_onnx = convert_sklearn(
            model, "lasso cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoCV-Dec4")

    def test_model_lasso_lars_int(self):
        model, X = fit_regression_model(LassoLars(), is_int=True)
        model_onnx = convert_sklearn(
            model, "lasso lars",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLarsInt-Dec4")

    def test_model_lasso_lars_bool(self):
        model, X = fit_regression_model(
            LassoLars(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "lasso lars",
            [("input", BooleanTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLarsBool")

    def test_model_multi_linear_regression(self):
        model, X = fit_regression_model(LinearRegression(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiLinearRegression-Dec4")

    def test_model_ard_regression(self):
        model, X = fit_regression_model(
            ARDRegression(), factor=0.001)
        model_onnx = convert_sklearn(
            model, "ard regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnARDRegression-Dec4")

    def test_model_theilsen(self):
        model, X = fit_regression_model(TheilSenRegressor())
        model_onnx = convert_sklearn(
            model, "thiel-sen regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnTheilSen-Dec4")

    def test_model_bayesian_ridge(self):
        model, X = fit_regression_model(BayesianRidge())
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBayesianRidge-Dec4")

    def test_model_bayesian_ridge_return_std(self):
        model, X = fit_regression_model(BayesianRidge(),
                                        n_features=2, n_samples=20)
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={BayesianRidge: {'return_std': True}})
        self.assertIsNotNone(model_onnx)

        sess = OnnxInference(model_onnx)
        outputs = sess.run({'input': X})
        pred, std = model.predict(X, return_std=True)
        self.assertEqualArray(pred, outputs['variable'].ravel(), decimal=4)
        self.assertEqualArray(std, outputs['std'].ravel(), decimal=4)

    def test_model_bayesian_ridge_return_std_double(self):
        model, X = fit_regression_model(BayesianRidge(),
                                        n_features=2, n_samples=100,
                                        n_informative=1)
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            options={BayesianRidge: {'return_std': True}})
        self.assertIsNotNone(model_onnx)

        X = X.astype(numpy.float64)
        sess = OnnxInference(model_onnx)
        outputs = sess.run({'input': X})
        pred, std = model.predict(X, return_std=True)
        self.assertEqualArray(pred, outputs['variable'].ravel())
        self.assertEqualArray(std, outputs['std'].ravel(), decimal=4)

    def test_model_bayesian_ridge_return_std_normalize(self):
        model, X = fit_regression_model(
            BayesianRidge(normalize=True),
            n_features=2, n_samples=50)
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={BayesianRidge: {'return_std': True}})
        self.assertIsNotNone(model_onnx)

        sess = OnnxInference(model_onnx)
        outputs = sess.run({'input': X})
        pred, std = model.predict(X, return_std=True)
        self.assertEqualArray(pred, outputs['variable'].ravel(), decimal=4)
        self.assertEqualArray(std, outputs['std'].ravel(), decimal=4)

    def test_model_bayesian_ridge_return_std_normalize_double(self):
        model, X = fit_regression_model(
            BayesianRidge(normalize=True),
            n_features=2, n_samples=50)
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            options={BayesianRidge: {'return_std': True}})
        self.assertIsNotNone(model_onnx)

        X = X.astype(numpy.float64)
        sess = OnnxInference(model_onnx)
        outputs = sess.run({'input': X})
        pred, std = model.predict(X, return_std=True)
        self.assertEqualArray(pred, outputs['variable'].ravel())
        self.assertEqualArray(std, outputs['std'].ravel(), decimal=4)

    def test_model_huber_regressor(self):
        model, X = fit_regression_model(HuberRegressor())
        model_onnx = convert_sklearn(
            model, "huber regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnHuberRegressor-Dec4")

    def test_model_multi_task_lasso(self):
        model, X = fit_regression_model(MultiTaskLasso(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "multi-task lasso",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiTaskLasso-Dec4")

    def test_model_multi_task_lasso_cv(self):
        model, X = fit_regression_model(MultiTaskLassoCV(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "mutli-task lasso cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiTaskLassoCV-Dec4")

    def test_model_multi_task_elasticnet(self):
        model, X = fit_regression_model(MultiTaskElasticNet(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "multi-task elasticnet",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiTaskElasticNet-Dec4")

    def test_model_orthogonal_matching_pursuit(self):
        model, X = fit_regression_model(
            OrthogonalMatchingPursuit())
        model_onnx = convert_sklearn(
            model, "orthogonal matching pursuit",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnOrthogonalMatchingPursuit-Dec4")

    def test_model_passive_aggressive_regressor(self):
        model, X = fit_regression_model(
            PassiveAggressiveRegressor())
        model_onnx = convert_sklearn(
            model, "passive aggressive regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnPassiveAggressiveRegressor-Dec4")

    def test_model_ransac_regressor_default(self):
        model, X = fit_regression_model(
            RANSACRegressor())
        model_onnx = convert_sklearn(
            model, "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnRANSACRegressor-Dec4")

    @ignore_warnings(ConvergenceWarning)
    def test_model_ransac_regressor_mlp(self):
        model, X = fit_regression_model(
            RANSACRegressor(
                base_estimator=MLPRegressor(solver='lbfgs')))
        model_onnx = convert_sklearn(
            model, "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnRANSACRegressorMLP-Dec3")

    def test_model_ransac_regressor_tree(self):
        model, X = fit_regression_model(
            RANSACRegressor(
                base_estimator=GradientBoostingRegressor()))
        model_onnx = convert_sklearn(
            model, "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnRANSACRegressorTree-Dec3")

    def test_model_multi_task_elasticnet_cv(self):
        model, X = fit_regression_model(MultiTaskElasticNetCV(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "multi-task elasticnet cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiTaskElasticNetCV-Dec4")

    def test_model_orthogonal_matching_pursuit_cv(self):
        model, X = fit_regression_model(
            OrthogonalMatchingPursuitCV())
        model_onnx = convert_sklearn(
            model, "orthogonal matching pursuit cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnOrthogonalMatchingPursuitCV-Dec4")


if __name__ == "__main__":
    unittest.main()
