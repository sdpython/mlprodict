"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
import numpy
from sklearn.base import ClusterMixin, BiclusterMixin, OutlierMixin
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSSVD
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, AdaBoostClassifier,
    BaggingClassifier, VotingClassifier, GradientBoostingClassifier
)
from sklearn.feature_selection import (
    RFE, RFECV, GenericUnivariateSelect,
    SelectPercentile, SelectFwe, SelectKBest,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import (
    ARDRegression, ElasticNetCV,
    LarsCV, LassoCV, LassoLarsCV, LassoLarsIC,
    SGDRegressor, OrthogonalMatchingPursuitCV,
    TheilSenRegressor, BayesianRidge, MultiTaskElasticNet,
    MultiTaskElasticNetCV, MultiTaskLassoCV, MultiTaskLasso,
    PassiveAggressiveClassifier, RidgeClassifier,
    RidgeClassifierCV, PassiveAggressiveRegressor,
    HuberRegressor, LogisticRegression, SGDClassifier,
    LogisticRegressionCV, Perceptron
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import (
    OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
)
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import (
    NearestCentroid, RadiusNeighborsClassifier,
    NeighborhoodComponentsAnalysis,
)
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC, LinearSVR, NuSVR, SVR, SVC, NuSVC
from skl2onnx.common.data_types import (
    FloatTensorType, DoubleTensorType
)


def _problem_for_predictor_binary_classification(dtype=numpy.float32):
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    binary classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    y[y == 2] = 1
    return (X, y, [('X', X[:1].astype(dtype))],
            'predict_proba', 1, X.astype(dtype))


def _problem_for_predictor_multi_classification(dtype=numpy.float32):
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    multi-class classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y, [('X', X[:1].astype(dtype))],
            'predict_proba', 1, X.astype(dtype))


def _problem_for_predictor_multi_classification_label(dtype=numpy.float32):
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    multi-class classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    y2 = numpy.zeros((y.shape[0], 3), dtype=numpy.int64)
    for i, _ in enumerate(y):
        y2[i, _] = 1
    for i in range(0, y.shape[0], 5):
        y2[i, (y[i] + 1) % 3] = 1
    return (X, y2, [('X', X[:1].astype(dtype))],
            'predict_proba', 1, X.astype(dtype))


def _problem_for_predictor_regression(many_output=False, options=None,
                                      nbfeat=None, nbrows=None, dtype=numpy.float32,
                                      **kwargs):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    regression problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target + numpy.arange(len(data.target)) / 100
    meth = 'predict' if kwargs is None else ('predict', kwargs)
    itt = [('X', X[:1].astype(dtype))]
    if nbfeat is not None:
        X = X[:, :nbfeat]
        itt = [('X', X[:1].astype(dtype))]
    if nbrows is not None:
        X = X[::nbrows, :]
        y = y[::nbrows]
        itt = [('X', X[:1].astype(dtype))]
    if options is not None:
        itt = itt, options
    return (X, y.astype(float), itt,
            meth, 'all' if many_output else 0, X.astype(dtype))


def _problem_for_predictor_multi_regression(many_output=False, options=None,
                                            nbfeat=None, nbrows=None,
                                            dtype=numpy.float32, **kwargs):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    multi-regression problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target.astype(float) + numpy.arange(len(data.target)) / 100
    meth = 'predict' if kwargs is None else ('predict', kwargs)
    itt = [('X', X[:1].astype(dtype))]
    if nbfeat is not None:
        X = X[:, :nbfeat]
        itt = [('X', X[:1].astype(dtype))]
    if nbrows is not None:
        X = X[::nbrows, :]
        y = y[::nbrows]
        itt = [('X', X[:1].astype(dtype))]
    if options is not None:
        itt = itt, options
    y2 = numpy.empty((y.shape[0], 2))
    y2[:, 0] = y
    y2[:, 1] = y + 0.5
    return (X, y2, itt,
            meth, 'all' if many_output else 0, X.astype(dtype))


def _problem_for_numerical_transform(dtype=numpy.float32):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', X[:1].astype(dtype))],
            'transform', 0, X.astype(dtype=numpy.float32))


def _problem_for_numerical_trainable_transform(dtype):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target + numpy.arange(len(data.target)) / 100
    return (X, y, [('X', X[:1].astype(dtype))],
            'transform', 0, X.astype(dtype))


def _problem_for_clustering(dtype=numpy.float32):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    clustering problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', X[:1].astype(dtype))],
            'predict', 0, X.astype(dtype))


def _problem_for_clustering_scores(dtype=numpy.float32):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    clustering problem, the score part, not the cluster.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', X[:1].astype(dtype))],
            'transform', 1, X.astype(dtype))


def _problem_for_outlier(dtype=numpy.float32):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', X[:1].astype(dtype))],
            'predict', 0, X.astype(dtype))


def _problem_for_numerical_scoring(dtype=numpy.float32):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target.astype(dtype) + numpy.arange(len(data.target)) / 100
    y /= numpy.max(y)
    return (X, y, [('X', X[:1].astype(dtype))],
            'score', 0, X.astype(dtype))


def _problem_for_clnoproba(dtype=numpy.float32):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y, [('X', X[:1].astype(dtype))],
            'predict', 0, X.astype(dtype))


def _problem_for_clnoproba_binary(dtype=numpy.float32):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem. Binary classification.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    y[y == 2] = 1
    return (X, y, [('X', X[:1].astype(dtype))],
            'predict', 0, X.astype(dtype))


def find_suitable_problem(model):
    """
    Determines problems suitable for a given
    :epkg:`scikit-learn` operator. It may be

    * `bin-class`: binary classification
    * `multi-class`: multi-class classification
    * `multi-label`: classification multi-label
      (multi label possible at the same time)
    * `regression`: regression
    * `multi-reg`: regression multi-output
    * `num-transform`: transform numerical features
    * `scoring`: transform numerical features, target is usually needed
    * `outlier`: outlier prediction
    * `linearsvc`: classifier without *predict_proba*
    * `cluster`: similar to transform
    * `num+y-trans`: similar to transform with targets
    * `num-trans-cluster`: similar to cluster, but returns
        scores or distances instead of cluster

    Suffix `nofit` indicates the predictions happens
    without the model being fitted. This is the case
    for :epkg:`sklearn:gaussian_process:GaussianProcessRegressor`.
    The suffix `-cov` indicates the method `predict` was called
    with parameter ``return_cov=True``, `-std` tells
    method `predict` was called with parameter ``return_std=True``.
    The suffix ``-noshapevar`` creates an input variable
    like the following ``[('X', FloatTensorType(["da", "db"]))]``.
    That's a way to bypass :epkg:`onnxruntime` shape checking
    as one part of the graph is designed to handle any
    kind of dimensions but apparently, if the input shape is
    precise, every part of the graph has to be precise. The strings
    used variables which means it is at the same time precise
    and unprecise. Suffix ``'-64'`` means the model will
    do double computations.

    The following script gives the list of :epkg:`scikit-learn`
    models and the problem they can be fitted on.

    .. runpython::
        :showcode:

        from mlprodict.onnxrt.validate import sklearn_operators, find_suitable_problem
        res = sklearn_operators()
        for model in res:
            try:
                prob = find_suitable_problem(model['cl'])
                print(model['name'], ":", prob)
            except RuntimeError:
                print("-", model['name'], ": no associated problem")
    """
    # Exceptions
    if model in {GaussianProcessRegressor}:
        return ['reg-nofit', 'multi-reg-nofit',
                'reg-nofit-cov', 'multi-reg-nofit-cov',
                'reg-nofit-std', 'multi-reg-nofit-std',
                'reg-noshapevar', 'multi-reg-noshapevar',
                'reg-cov', 'multi-reg-cov',
                'reg-std-d2-noshapevar', 'mreg-std-d2-noshapevar',
                'regression', 'regression-64']

    if model in {BaggingClassifier, BernoulliNB, CalibratedClassifierCV,
                 ComplementNB, GaussianNB, GaussianProcessClassifier,
                 GradientBoostingClassifier, LabelPropagation, LabelSpreading,
                 LinearDiscriminantAnalysis, LogisticRegressionCV,
                 MultinomialNB, NuSVC, Perceptron, QuadraticDiscriminantAnalysis,
                 RandomizedSearchCV, SGDClassifier, SVC}:
        return ['bin-class', 'multi-class']

    if model in {AdaBoostRegressor}:
        return ['regression']

    if model in {LinearSVC, NearestCentroid}:
        return ['clnoproba', 'clnoproba-64']

    if model in {RFE, RFECV}:
        return ['bin-class', 'multi-class', 'regression']

    if model in {GridSearchCV}:
        return ['bin-class', 'multi-class',
                'regression', 'multi-reg',
                'cluster', 'outlier', 'multi-label']

    if model in {VotingClassifier}:
        return ['bin-class']

    # specific scenarios
    if model in {IsotonicRegression}:
        return ['num+y-trans', 'regression']

    if model in {ARDRegression, BayesianRidge, ElasticNetCV,
                 GradientBoostingRegressor,
                 LarsCV, LassoCV, LassoLarsCV, LassoLarsIC,
                 LinearSVR, NuSVR, OrthogonalMatchingPursuitCV,
                 PassiveAggressiveRegressor, SGDRegressor,
                 TheilSenRegressor, HuberRegressor, SVR}:
        return ['regression', 'regression-64']

    if model in {MultiOutputClassifier}:
        return ['multi-class', 'multi-label']

    if model in {MultiOutputRegressor, MultiTaskElasticNet,
                 MultiTaskElasticNetCV, MultiTaskLassoCV,
                 MultiTaskLasso}:
        return ['multi-reg']

    if model in {OneVsOneClassifier, OutputCodeClassifier,
                 PassiveAggressiveClassifier, RadiusNeighborsClassifier,
                 RidgeClassifier, RidgeClassifierCV}:
        return ['binclnoproba', 'clnoproba']

    # trainable transform
    if model in {GenericUnivariateSelect,
                 NeighborhoodComponentsAnalysis,
                 PLSSVD, SelectFwe, SelectKBest,
                 SelectPercentile}:
        return ["num+y-trans"]

    # no multi-label
    if model in {AdaBoostClassifier, LogisticRegression}:
        return ['bin-class', 'bin-class-64', 'multi-class']

    # predict, predict_proba
    if hasattr(model, 'predict_proba'):
        if model is OneVsRestClassifier:
            return ['multi-class', 'multi-label']
        else:
            return ['bin-class', 'multi-class', 'multi-label']

    if hasattr(model, 'predict'):
        if "Classifier" in str(model):
            return ['bin-class', 'bin-class-64', 'multi-class', 'multi-label']
        elif "Regressor" in str(model):
            return ['regression', 'multi-reg', 'regression-64']

    # Generic case.
    res = []
    if hasattr(model, 'transform'):
        if issubclass(model, (RegressorMixin, ClassifierMixin)):
            res.extend(['num+y-trans'])
        elif issubclass(model, (ClusterMixin, BiclusterMixin)):
            res.extend(['num-trans-cluster'])
        else:
            res.extend(['num-transform'])

    if hasattr(model, 'predict') and issubclass(model, (ClusterMixin, BiclusterMixin)):
        res.extend(['cluster'])

    if issubclass(model, (OutlierMixin)):
        res.extend(['outlier'])

    if issubclass(model, ClassifierMixin):
        res.extend(['bin-class', 'bin-class-64', 'multi-class', 'multi-label'])
    if issubclass(model, RegressorMixin):
        res.extend(['regression', 'multi-reg', 'regression-64'])

    if len(res) == 0 and hasattr(model, 'fit') and hasattr(model, 'score'):
        return ['scoring']
    if len(res) > 0:
        return res

    raise RuntimeError("Unable to find problem for model '{}' - {}."
                       "".format(model.__name__, model.__bases__))


def _guess_noshape(obj, shape):
    if isinstance(obj, numpy.ndarray):
        if obj.dtype == numpy.float32:
            return FloatTensorType(shape)
        elif obj.dtype == numpy.float64:
            return DoubleTensorType(shape)
        else:
            raise NotImplementedError(
                "Unable to process object(1) [{}].".format(obj))
    else:
        raise NotImplementedError(
            "Unable to process object(2) [{}].".format(obj))


def _noshapevar(fct):

    def process_itt(itt, Xort):
        if isinstance(itt, tuple):
            return (process_itt(itt[0], Xort), itt[1])
        else:
            name = "V%s_" % str(id(Xort))[:5]
            new_itt = []
            for i, (a, b) in enumerate(itt):
                shape = [name + str(i) for s in b.shape]
                new_itt.append((a, _guess_noshape(b, shape)))
            return new_itt

    def new_fct(**kwargs):
        X, y, itt, meth, mo, Xort = fct(**kwargs)
        new_itt = process_itt(itt, Xort)
        return X, y, new_itt, meth, mo, Xort
    return new_fct


_problems = {
    "bin-class": _problem_for_predictor_binary_classification,
    "bin-class-64": lambda: _problem_for_predictor_binary_classification(dtype=numpy.float64),
    "multi-class": _problem_for_predictor_multi_classification,
    "regression": _problem_for_predictor_regression,
    "regression-64": lambda: _problem_for_predictor_regression(dtype=numpy.float64),
    "multi-reg": _problem_for_predictor_multi_regression,
    "multi-label": _problem_for_predictor_multi_classification_label,
    "num-transform": _problem_for_numerical_transform,
    "scoring": _problem_for_numerical_scoring,
    'outlier': _problem_for_outlier,
    'clnoproba': _problem_for_clnoproba,
    'binclnoproba': _problem_for_clnoproba_binary,
    'cluster': _problem_for_clustering,
    'num-trans-cluster': _problem_for_clustering_scores,
    'num+y-trans': _problem_for_numerical_trainable_transform,
    #
    "bin-class-nofit": (lambda: _problem_for_predictor_binary_classification() + (False, )),
    "multi-class-nofit": (lambda: _problem_for_predictor_multi_classification() + (False, )),
    "reg-nofit": (lambda: _problem_for_predictor_regression() + (False, )),
    "multi-reg-nofit": (lambda: _problem_for_predictor_multi_regression() + (False, )),
    #
    "reg-nofit-cov": (lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}}, return_cov=True) + (False, )),
    "multi-reg-nofit-cov": (lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}}, return_cov=True) + (False, )),
    #
    "reg-nofit-std": (lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}}, return_std=True) + (False, )),
    "multi-reg-nofit-std": (lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}}, return_std=True) + (False, )),
    #
    "reg-cov": (lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}}, return_cov=True)),
    "multi-reg-cov": (lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}}, return_cov=True)),
    #
    "reg-std-d2": (lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, nbfeat=2, nbrows=10)),
    "multi-reg-std-d2": (lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, nbfeat=2, nbrows=10)),
    #
    'reg-noshapevar': _noshapevar(_problem_for_predictor_regression),
    'multi-reg-noshapevar': _noshapevar(_problem_for_predictor_multi_regression),
    "reg-std-d2-noshapevar": (_noshapevar(lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, nbfeat=2, nbrows=10))),
    "mreg-std-d2-noshapevar": (_noshapevar(lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, nbfeat=2, nbrows=10))),
}
