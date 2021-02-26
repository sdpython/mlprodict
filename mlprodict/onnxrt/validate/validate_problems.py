"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
import numpy
from sklearn.base import (
    ClusterMixin, BiclusterMixin, OutlierMixin,
    RegressorMixin, ClassifierMixin)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSSVD
from sklearn.datasets import load_iris
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, AdaBoostClassifier,
    BaggingClassifier, VotingClassifier, GradientBoostingClassifier,
    RandomForestClassifier)
try:
    from sklearn.ensemble import StackingClassifier, StackingRegressor
except ImportError:  # pragma: no cover
    # new in 0.22
    StackingClassifier, StackingRegressor = None, None
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer, TfidfTransformer)
from sklearn.experimental import enable_hist_gradient_boosting  # pylint: disable=W0611
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier)
from sklearn.feature_selection import (
    RFE, RFECV, GenericUnivariateSelect,
    SelectPercentile, SelectFwe, SelectKBest,
    SelectFdr, SelectFpr, SelectFromModel)
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
    LogisticRegressionCV, Perceptron)
from sklearn.mixture._base import BaseMixture
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import (
    OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier)
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import (
    NearestCentroid, RadiusNeighborsClassifier,
    NeighborhoodComponentsAnalysis)
from sklearn.preprocessing import (
    LabelBinarizer, LabelEncoder,
    OneHotEncoder, PowerTransformer)
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC, LinearSVR, NuSVR, SVR, SVC, NuSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import shuffle
from skl2onnx.common.data_types import (
    FloatTensorType, DoubleTensorType, StringTensorType, DictionaryType)
from ._validate_problems_helper import (
    _noshapevar, _1d_problem, text_alpha_num)


def _modify_dimension(X, n_features, seed=19):
    """
    Modifies the number of features to increase
    or reduce the number of features.

    @param      X           features matrix
    @param      n_features  number of features
    @param      seed        random seed (to get the same
                            dataset at each call)
    @return                 new featurs matrix
    """
    if n_features is None or n_features == X.shape[1]:
        return X
    if n_features < X.shape[1]:
        return X[:, :n_features]
    rstate = numpy.random.RandomState(seed)  # pylint: disable=E1101
    res = numpy.empty((X.shape[0], n_features), dtype=X.dtype)
    res[:, :X.shape[1]] = X[:, :]
    div = max((n_features // X.shape[1]) + 1, 2)
    for i in range(X.shape[1], res.shape[1]):
        j = i % X.shape[1]
        col = X[:, j]
        if X.dtype in (numpy.float32, numpy.float64):
            sigma = numpy.var(col) ** 0.5
            rnd = rstate.randn(len(col)) * sigma / div
            col2 = col + rnd
            res[:, j] -= col2 / div
            res[:, i] = col2
        elif X.dtype in (numpy.int32, numpy.int64):
            perm = rstate.permutation(col)
            h = rstate.randint(0, div) % X.shape[0]
            col2 = col.copy()
            col2[h::div] = perm[h::div]  # pylint: disable=E1136
            res[:, i] = col2
            h = (h + 1) % X.shape[0]
            res[h, j] = perm[h]  # pylint: disable=E1136
        else:  # pragma: no cover
            raise NotImplementedError(  # pragma: no cover
                "Unable to add noise to a feature for this type {}".format(X.dtype))
    return res


###########
# datasets
###########


def _problem_for_predictor_binary_classification(
        dtype=numpy.float32, n_features=None, add_nan=False):
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    binary classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target
    y[y == 2] = 1
    if add_nan:
        rows = numpy.random.randint(0, X.shape[0] - 1, X.shape[0] // 3)
        cols = numpy.random.randint(0, X.shape[1] - 1, X.shape[0] // 3)
        X[rows, cols] = numpy.nan
    X = X.astype(dtype)
    y = y.astype(numpy.int64)
    return (X, y, [('X', X[:1].astype(dtype))],
            'predict_proba', 1, X.astype(dtype))


def _problem_for_predictor_multi_classification(dtype=numpy.float32, n_features=None):
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    m-cl classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target
    X = X.astype(dtype)
    y = y.astype(numpy.int64)
    return (X, y, [('X', X[:1].astype(dtype))],
            'predict_proba', 1, X.astype(dtype))


def _problem_for_mixture(dtype=numpy.float32, n_features=None):
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    m-cl classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target
    X = X.astype(dtype)
    y = y.astype(numpy.int64)
    return (X, None, [('X', X[:1].astype(dtype))],
            'predict_proba', 1, X.astype(dtype))


def _problem_for_predictor_multi_classification_label(dtype=numpy.float32, n_features=None):
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    m-cl classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target
    y2 = numpy.zeros((y.shape[0], 3), dtype=numpy.int64)
    for i, _ in enumerate(y):
        y2[i, _] = 1
    for i in range(0, y.shape[0], 5):
        y2[i, (y[i] + 1) % 3] = 1
    X = X.astype(dtype)
    y2 = y2.astype(numpy.int64)
    return (X, y2, [('X', X[:1].astype(dtype))],
            'predict_proba', 1, X.astype(dtype))


def _problem_for_predictor_regression(many_output=False, options=None,
                                      n_features=None, nbrows=None,
                                      dtype=numpy.float32, add_nan=False,
                                      **kwargs):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    regression problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target + numpy.arange(len(data.target)) / 100
    meth = 'predict' if kwargs is None else ('predict', kwargs)
    itt = [('X', X[:1].astype(dtype))]
    if n_features is not None:
        X = X[:, :n_features]
        itt = [('X', X[:1].astype(dtype))]
    if nbrows is not None:
        X = X[:nbrows, :]
        y = y[:nbrows]
        itt = [('X', X[:1].astype(dtype))]
    if options is not None:
        itt = itt, options
    if add_nan:
        rows = numpy.random.randint(0, X.shape[0] - 1, X.shape[0] // 3)
        cols = numpy.random.randint(0, X.shape[1] - 1, X.shape[0] // 3)
        X[rows, cols] = numpy.nan
    X = X.astype(dtype)
    y = y.astype(dtype)
    return (X, y, itt,
            meth, 'all' if many_output else 0, X.astype(dtype))


def _problem_for_predictor_multi_regression(many_output=False, options=None,
                                            n_features=None, nbrows=None,
                                            dtype=numpy.float32, **kwargs):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    mregression problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target.astype(float) + numpy.arange(len(data.target)) / 100
    meth = 'predict' if kwargs is None else ('predict', kwargs)
    itt = [('X', X[:1].astype(dtype))]
    if n_features is not None:
        X = X[:, :n_features]
        itt = [('X', X[:1].astype(dtype))]
    if nbrows is not None:
        X = X[:nbrows, :]
        y = y[:nbrows]
        itt = [('X', X[:1].astype(dtype))]
    if options is not None:
        itt = itt, options
    y2 = numpy.empty((y.shape[0], 2))
    y2[:, 0] = y
    y2[:, 1] = y + 0.5
    X = X.astype(dtype)
    y2 = y2.astype(dtype)
    return (X, y2, itt,
            meth, 'all' if many_output else 0, X.astype(dtype))


def _problem_for_numerical_transform(dtype=numpy.float32, n_features=None):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    X = X.astype(dtype)
    return (X, None, [('X', X[:1].astype(dtype))],
            'transform', 0, X.astype(dtype=numpy.float32))


def _problem_for_numerical_transform_positive(dtype=numpy.float32, n_features=None):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*data.data.shape) / 3
    X = numpy.abs(data.data + rnd)
    X = _modify_dimension(X, n_features)
    X = X.astype(dtype)
    return (X, None, [('X', X[:1].astype(dtype))],
            'transform', 0, X.astype(dtype=numpy.float32))


def _problem_for_numerical_trainable_transform(dtype=numpy.float32, n_features=None):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target + numpy.arange(len(data.target)) / 100
    X = X.astype(dtype)
    y = y.astype(dtype)
    return (X, y, [('X', X[:1].astype(dtype))],
            'transform', 0, X.astype(dtype))


def _problem_for_numerical_trainable_transform_cl(dtype=numpy.float32, n_features=None):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target
    X = X.astype(dtype)
    y = y.astype(numpy.int64)
    return (X, y, [('X', X[:1].astype(dtype))],
            'transform', 0, X.astype(dtype))


def _problem_for_clustering(dtype=numpy.float32, n_features=None):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    clustering problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    X = X.astype(dtype)
    return (X, None, [('X', X[:1].astype(dtype))],
            'predict', 0, X.astype(dtype))


def _problem_for_clustering_scores(dtype=numpy.float32, n_features=None):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    clustering problem, the score part, not the cluster.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    X = X.astype(dtype)
    return (X, None, [('X', X[:1].astype(dtype))],
            'transform', 1, X.astype(dtype))


def _problem_for_outlier(dtype=numpy.float32, n_features=None):
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    X = X.astype(dtype)
    return (X, None, [('X', X[:1].astype(dtype))],
            'predict', 0, X.astype(dtype))


def _problem_for_numerical_scoring(dtype=numpy.float32, n_features=None):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    y = data.target.astype(dtype) + numpy.arange(len(data.target)) / 100
    y /= numpy.max(y)
    X = X.astype(dtype)
    y = y.astype(dtype)
    return (X, y, [('X', X[:1].astype(dtype))],
            'score', 0, X.astype(dtype))


def _problem_for_clnoproba(dtype=numpy.float32, n_features=None):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target
    X = X.astype(dtype)
    y = y.astype(numpy.int64)
    return (X, y, [('X', X[:1].astype(dtype))],
            'predict', 0, X.astype(dtype))


def _problem_for_clnoproba_binary(dtype=numpy.float32, n_features=None, add_nan=False):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem. Binary classification.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target
    y[y == 2] = 1
    if add_nan:
        rows = numpy.random.randint(0, X.shape[0] - 1, X.shape[0] // 3)
        cols = numpy.random.randint(0, X.shape[1] - 1, X.shape[0] // 3)
        X[rows, cols] = numpy.nan
    X = X.astype(dtype)
    y = y.astype(numpy.int64)
    return (X, y, [('X', X[:1].astype(dtype))],
            'predict', 0, X.astype(dtype))


def _problem_for_cl_decision_function(dtype=numpy.float32, n_features=None):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target
    X = X.astype(dtype)
    y = y.astype(numpy.int64)
    return (X, y, [('X', X[:1].astype(dtype))],
            'decision_function', 1, X.astype(dtype))


def _problem_for_cl_decision_function_binary(dtype=numpy.float32, n_features=None):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem. Binary classification.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*X.shape) / 3
    X += rnd
    X = _modify_dimension(X, n_features)
    y = data.target
    y[y == 2] = 1
    X = X.astype(dtype)
    y = y.astype(numpy.int64)
    return (X, y, [('X', X[:1].astype(dtype))],
            'decision_function', 1, X.astype(dtype))


def _problem_for_label_encoder(dtype=numpy.int64, n_features=None):
    """
    Returns a problem for the :epkg:`sklearn:preprocessing:LabelEncoder`.
    """
    data = load_iris()
    # X = data.data
    y = data.target.astype(dtype)
    itt = [('X', y[:1].astype(dtype))]
    y = y.astype(dtype)
    return (y, None, itt, 'transform', 0, y)


def _problem_for_dict_vectorizer(dtype=numpy.float32, n_features=None):
    """
    Returns a problem for the :epkg:`sklearn:feature_extraction:DictVectorizer`.
    """
    data = load_iris()
    # X = data.data
    y = data.target
    y2 = [{_: dtype(1000 + i)} for i, _ in enumerate(y)]
    y2[0][2] = -2
    cltype = FloatTensorType if dtype == numpy.float32 else DoubleTensorType
    itt = [("X", DictionaryType(StringTensorType([1]), cltype([1])))]
    y2 = numpy.array(y2)
    y = y.astype(numpy.int64)
    return (y2, y, itt, 'transform', 0, y2)


def _problem_for_tfidf_vectorizer(dtype=numpy.float32, n_features=None):
    """
    Returns a problem for the :epkg:`sklearn:feature_extraction:text:TfidfVectorizer`.
    """
    X = numpy.array([_[0] for _ in text_alpha_num])
    y = numpy.array([_[1] for _ in text_alpha_num], dtype=dtype)
    itt = [("X", StringTensorType([None]))]
    return (X, y, itt, 'transform', 0, X)


def _problem_for_tfidf_transformer(dtype=numpy.float32, n_features=None):
    """
    Returns a problem for the :epkg:`sklearn:feature_extraction:text:TfidfTransformer`.
    """
    X = numpy.array([_[0] for _ in text_alpha_num])
    y = numpy.array([_[1] for _ in text_alpha_num], dtype=dtype)
    X2 = CountVectorizer().fit_transform(X).astype(dtype)
    cltype = FloatTensorType if dtype == numpy.float32 else DoubleTensorType
    itt = [("X", cltype([None, X2.shape[1]]))]
    return (X2, y, itt, 'transform', 0, X2)


def _problem_for_feature_hasher(dtype=numpy.float32, n_features=None):
    """
    Returns a problem for the :epkg:`sklearn:feature_extraction:DictVectorizer`.
    """
    data = load_iris()
    # X = data.data
    y = data.target
    y2 = [{("cl%d" % _): dtype(1000 + i)} for i, _ in enumerate(y)]
    y2[0]["cl2"] = -2
    cltype = FloatTensorType if dtype == numpy.float32 else DoubleTensorType
    itt = [("X", DictionaryType(StringTensorType([1]), cltype([1])))]
    y2 = numpy.array(y2)
    return (y2, y, itt, 'transform', 0, y2)


def _problem_for_one_hot_encoder(dtype=numpy.float32, n_features=None):
    """
    Returns a problem for the :epkg:`sklearn:preprocessing:OneHotEncoder`.
    """
    data = load_iris()
    state = numpy.random.RandomState(seed=34)  # pylint: disable=E1101
    rnd = state.randn(*data.data.shape) / 3
    X = _modify_dimension(data.data + rnd, n_features)
    X = X.astype(numpy.int32).astype(dtype)
    y = data.target
    X, y = shuffle(X, y, random_state=1)
    itt = [('X', X[:1].astype(dtype))]
    return (X[:, :1], y, itt, 'transform', 0, X[:, :1].astype(dtype))


def find_suitable_problem(model):
    """
    Determines problems suitable for a given
    :epkg:`scikit-learn` operator. It may be

    * `b-cl`: binary classification
    * `m-cl`: m-cl classification
    * `m-label`: classification m-label
      (multiple labels possible at the same time)
    * `reg`: regression
    * `m-reg`: regression multi-output
    * `num-tr`: transform numerical features
    * `num-tr-pos`: transform numerical positive features
    * `scoring`: transform numerical features, target is usually needed
    * `outlier`: outlier prediction
    * `linearsvc`: classifier without *predict_proba*
    * `cluster`: similar to transform
    * `num+y-tr`: similar to transform with targets
    * `num+y-tr-cl`: similar to transform with classes
    * `num-tr-clu`: similar to cluster, but returns
        scores or distances instead of cluster
    * `key-col`: list of dictionaries
    * `text-col`: one column of text

    Suffix `nofit` indicates the predictions happens
    without the model being fitted. This is the case
    for :epkg:`sklearn:gaussian_process:GaussianProcessRegressor`.
    The suffix `-cov` indicates the method `predict` was called
    with parameter ``return_cov=True``, `-std` tells
    method `predict` was called with parameter ``return_std=True``.
    The suffix ``-NSV`` creates an input variable
    like the following ``[('X', FloatTensorType([None, None]))]``.
    That's a way to bypass :epkg:`onnxruntime` shape checking
    as one part of the graph is designed to handle any
    kind of dimensions but apparently, if the input shape is
    precise, every part of the graph has to be precise. The strings
    used variables which means it is at the same time precise
    and unprecise. Suffix ``'-64'`` means the model will
    do double computations. Suffix ``-nop`` means the classifier
    does not implement method *predict_proba*. Suffix ``-1d``
    means a one dimension problem (one feature). Suffix ``-dec``
    checks method `decision_function`.

    The following script gives the list of :epkg:`scikit-learn`
    models and the problem they can be fitted on.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning
        :rst:

        from mlprodict.onnxrt.validate.validate import (
            sklearn_operators, find_suitable_problem)
        from pyquickhelper.pandashelper import df2rst
        from pandas import DataFrame
        res = sklearn_operators()
        rows = []
        for model in res[:20]:
            name = model['name']
            row = dict(name=name)
            try:
                prob = find_suitable_problem(model['cl'])
                if prob is None:
                    continue
                for p in prob:
                    row[p] = 'X'
            except RuntimeError:
                pass
            rows.append(row)
        df = DataFrame(rows).set_index('name')
        df = df.sort_index()
        print(df2rst(df, index=True))

    The list is truncated. The full list can be found at
    :ref:`l-model-problem-list`.
    """
    from ...onnx_conv.validate_scenarios import find_suitable_problem as ext_find_suitable_problem

    def _internal(model):  # pylint: disable=R0911

        # checks that this model is not overwritten by this module
        ext = ext_find_suitable_problem(model)
        if ext is not None:
            return ext

        # Exceptions
        if model in {GaussianProcessRegressor}:
            # m-reg causes MemoryError on some machine.
            return ['~b-reg-NF-64',  # '~m-reg-NF-64',
                    '~b-reg-NF-cov-64',  # '~m-reg-NF-cov-64',
                    '~b-reg-NF-std-64',  # '~m-reg-NF-std-64',
                    '~b-reg-NSV-64',  # '~m-reg-NSV-64',
                    '~b-reg-cov-64',  # '~m-reg-cov-64',
                    '~b-reg-std-NSV-64',  # '~m-reg-std-NSV-64',
                    'b-reg', '~b-reg-64',  # 'm-reg'
                    ]

        if model in {DictVectorizer}:
            return ['key-int-col']

        if model in {TfidfVectorizer, CountVectorizer}:
            return ['text-col']

        if model in {TfidfTransformer}:
            return ['bow']

        if model in {FeatureHasher}:
            return ['key-str-col']

        if model in {OneHotEncoder}:
            return ['one-hot']

        if model in {LabelBinarizer, LabelEncoder}:
            return ['int-col']

        if model in {NuSVC, SVC, SGDClassifier,
                     HistGradientBoostingClassifier}:
            return ['b-cl', 'm-cl', '~b-cl-64', '~b-cl-nan']

        if model in {GaussianProcessClassifier}:
            return ['b-cl', 'm-cl', '~b-cl-64']

        if model in {BaggingClassifier, BernoulliNB, CalibratedClassifierCV,
                     ComplementNB, GaussianNB,
                     GradientBoostingClassifier, LabelPropagation, LabelSpreading,
                     LinearDiscriminantAnalysis, LogisticRegressionCV,
                     MultinomialNB, QuadraticDiscriminantAnalysis,
                     RandomizedSearchCV}:
            return ['b-cl', 'm-cl']

        if model in {Perceptron}:
            return ['~b-cl-nop', '~m-cl-nop', '~b-cl-dec', '~m-cl-dec']

        if model in {AdaBoostRegressor}:
            return ['b-reg', '~b-reg-64']

        if model in {HistGradientBoostingRegressor}:
            return ['b-reg', '~b-reg-64', '~b-reg-nan', '~b-reg-nan-64']

        if model in {LinearSVC, NearestCentroid}:
            return ['~b-cl-nop', '~b-cl-nop-64']

        if model in {RFE, RFECV}:
            return ['num+y-tr']

        if model in {GridSearchCV}:
            return ['b-cl', 'm-cl',
                    'b-reg', 'm-reg',
                    '~b-reg-64', '~b-cl-64',
                    'cluster', 'outlier', '~m-label']

        if model in {VotingClassifier}:
            return ['b-cl', 'm-cl']

        if StackingClassifier is not None and model in {StackingClassifier}:
            return ['b-cl']

        if StackingRegressor is not None and model in {StackingRegressor}:
            return ['b-reg']

        # specific scenarios
        if model in {IsotonicRegression}:
            return ['~num+y-tr-1d', '~b-reg-1d']

        if model in {ARDRegression, BayesianRidge, ElasticNetCV,
                     GradientBoostingRegressor,
                     LarsCV, LassoCV, LassoLarsCV, LassoLarsIC,
                     LinearSVR, NuSVR, OrthogonalMatchingPursuitCV,
                     PassiveAggressiveRegressor, SGDRegressor,
                     TheilSenRegressor, HuberRegressor, SVR}:
            return ['b-reg', '~b-reg-64']

        if model in {MultiOutputClassifier}:
            return ['m-cl', '~m-label']

        if model in {MultiOutputRegressor, MultiTaskElasticNet,
                     MultiTaskElasticNetCV, MultiTaskLassoCV,
                     MultiTaskLasso}:
            return ['m-reg']

        if model in {OneVsOneClassifier, OutputCodeClassifier,
                     PassiveAggressiveClassifier, RadiusNeighborsClassifier}:
            return ['~b-cl-nop', '~m-cl-nop']

        if model in {RidgeClassifier, RidgeClassifierCV}:
            return ['~b-cl-nop', '~m-cl-nop', '~m-label']

        # trainable transform
        if model in {GenericUnivariateSelect,
                     NeighborhoodComponentsAnalysis,
                     PLSSVD, SelectKBest,
                     SelectPercentile, SelectFromModel}:
            return ["num+y-tr"]

        if model in {SelectFwe, SelectFdr, SelectFpr}:
            return ["num+y-tr-cl"]

        # no m-label
        if model in {AdaBoostClassifier}:
            return ['b-cl', '~b-cl-64', 'm-cl']

        if model in {LogisticRegression}:
            return ['b-cl', '~b-cl-64', 'm-cl', '~b-cl-dec', '~m-cl-dec']

        if model in {RandomForestClassifier}:
            return ['b-cl', '~b-cl-64', 'm-cl', '~m-label']

        if model in {DecisionTreeClassifier, ExtraTreeClassifier}:
            return ['b-cl', '~b-cl-64', 'm-cl', '~b-cl-f100', '~m-label']

        if model in {DecisionTreeRegressor}:
            return ['b-reg', 'm-reg', '~b-reg-64', '~m-reg-64', '~b-reg-f100']

        if model in {LatentDirichletAllocation, NMF, PowerTransformer}:
            return ['num-tr-pos']

        if hasattr(model, 'predict'):
            if "Classifier" in str(model):
                return ['b-cl', '~b-cl-64', 'm-cl', '~m-label']
            elif "Regressor" in str(model):
                return ['b-reg', 'm-reg', '~b-reg-64', '~m-reg-64']

        # Generic case.
        res = []
        if hasattr(model, 'transform'):
            if issubclass(model, (RegressorMixin, ClassifierMixin)):
                res.extend(['num+y-tr'])
            elif issubclass(model, (ClusterMixin, BiclusterMixin)):
                res.extend(['~num-tr-clu', '~num-tr-clu-64'])
            else:
                res.extend(['num-tr'])

        if hasattr(model, 'predict') and issubclass(model, (ClusterMixin, BiclusterMixin)):
            res.extend(['cluster', '~b-clu-64'])

        if issubclass(model, (OutlierMixin)):
            res.extend(['outlier'])

        if issubclass(model, ClassifierMixin):
            if model is OneVsRestClassifier:
                return ['m-cl', '~m-label']
            res.extend(['b-cl', '~b-cl-64', 'm-cl', '~m-label'])
        if issubclass(model, RegressorMixin):
            res.extend(['b-reg', 'm-reg', '~b-reg-64', '~m-reg-64'])
        if issubclass(model, BaseMixture):
            res.extend(['mix', '~mix-64'])

        if len(res) > 0:
            return res

        raise RuntimeError("Unable to find problem for model '{}' - {}."
                           "".format(model.__name__, model.__bases__))

    res = _internal(model)
    for r in res:
        if r not in _problems:
            raise ValueError(  # pragma: no cover
                "Unrecognized problem '{}' in\n{}".format(
                    r, "\n".join(sorted(_problems))))
    return res


_problems = {
    # standard
    "b-cl": _problem_for_predictor_binary_classification,
    "m-cl": _problem_for_predictor_multi_classification,
    "b-reg": _problem_for_predictor_regression,
    "m-reg": _problem_for_predictor_multi_regression,
    "num-tr": _problem_for_numerical_transform,
    "num-tr-pos": _problem_for_numerical_transform_positive,
    'outlier': _problem_for_outlier,
    'cluster': _problem_for_clustering,
    'num+y-tr': _problem_for_numerical_trainable_transform,
    'num+y-tr-cl': _problem_for_numerical_trainable_transform_cl,
    'mix': _problem_for_mixture,
    # others
    '~num-tr-clu': _problem_for_clustering_scores,
    "~m-label": _problem_for_predictor_multi_classification_label,
    "~scoring": _problem_for_numerical_scoring,
    '~b-cl-nop': _problem_for_clnoproba_binary,
    '~m-cl-nop': _problem_for_clnoproba,
    '~b-cl-dec': _problem_for_cl_decision_function_binary,
    '~m-cl-dec': _problem_for_cl_decision_function,
    # nan
    "~b-reg-nan": lambda n_features=None: _problem_for_predictor_regression(
        n_features=n_features, add_nan=True),
    "~b-reg-nan-64": lambda n_features=None: _problem_for_predictor_regression(
        dtype=numpy.float64, n_features=n_features, add_nan=True),
    "~b-cl-nan": lambda dtype=numpy.float32, n_features=None: _problem_for_predictor_binary_classification(
        dtype=dtype, n_features=n_features, add_nan=True),
    # 100 features
    "~b-reg-f100": lambda n_features=100: _problem_for_predictor_regression(
        n_features=n_features or 100),
    "~b-cl-f100": lambda n_features=100: _problem_for_predictor_binary_classification(
        n_features=n_features or 100),
    # 64
    "~b-cl-64": lambda n_features=None: _problem_for_predictor_binary_classification(
        dtype=numpy.float64, n_features=n_features),
    "~b-reg-64": lambda n_features=None: _problem_for_predictor_regression(
        dtype=numpy.float64, n_features=n_features),
    '~b-cl-nop-64': lambda n_features=None: _problem_for_clnoproba(
        dtype=numpy.float64, n_features=n_features),
    '~b-clu-64': lambda n_features=None: _problem_for_clustering(
        dtype=numpy.float64, n_features=n_features),
    '~b-cl-dec-64': lambda n_features=None: _problem_for_cl_decision_function_binary(
        dtype=numpy.float64, n_features=n_features),
    '~num-tr-clu-64': lambda n_features=None: _problem_for_clustering_scores(
        dtype=numpy.float64, n_features=n_features),
    "~m-reg-64": lambda n_features=None: _problem_for_predictor_multi_regression(
        dtype=numpy.float64, n_features=n_features),
    "~num-tr-64": lambda n_features=None: _problem_for_numerical_transform(
        dtype=numpy.float64, n_features=n_features),
    '~mix-64': lambda n_features=None: _problem_for_mixture(
        dtype=numpy.float64, n_features=n_features),
    #
    "~b-cl-NF": (lambda n_features=None: _problem_for_predictor_binary_classification(
        n_features=n_features) + (False, )),
    "~m-cl-NF": (lambda n_features=None: _problem_for_predictor_multi_classification(
        n_features=n_features) + (False, )),
    "~b-reg-NF": (lambda n_features=None: _problem_for_predictor_regression(
        n_features=n_features) + (False, )),
    "~m-reg-NF": (lambda n_features=None: _problem_for_predictor_multi_regression(
        n_features=n_features) + (False, )),
    #
    "~b-cl-NF-64": (lambda n_features=None: _problem_for_predictor_binary_classification(
        dtype=numpy.float64, n_features=n_features) + (False, )),
    "~m-cl-NF-64": (lambda n_features=None: _problem_for_predictor_multi_classification(
        dtype=numpy.float64, n_features=n_features) + (False, )),
    "~b-reg-NF-64": (lambda n_features=None: _problem_for_predictor_regression(
        dtype=numpy.float64, n_features=n_features) + (False, )),
    "~m-reg-NF-64": (lambda n_features=None: _problem_for_predictor_multi_regression(
        dtype=numpy.float64, n_features=n_features) + (False, )),
    # GaussianProcess
    "~b-reg-NF-cov-64": (lambda n_features=None: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}},
        return_cov=True, dtype=numpy.float64, n_features=n_features) + (False, )),
    "~m-reg-NF-cov-64": (lambda n_features=None: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}},
        return_cov=True, dtype=numpy.float64, n_features=n_features) + (False, )),
    #
    "~b-reg-NF-std-64": (lambda n_features=None: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64, n_features=n_features) + (False, )),
    "~m-reg-NF-std-64": (lambda n_features=None: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64, n_features=n_features) + (False, )),
    #
    "~b-reg-cov-64": (lambda n_features=None: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}},
        return_cov=True, dtype=numpy.float64, n_features=n_features)),
    "~m-reg-cov-64": (lambda n_features=None: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}},
        return_cov=True, dtype=numpy.float64, n_features=n_features)),
    #
    "~reg-std-64": (lambda n_features=None: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64, n_features=n_features)),
    "~m-reg-std-64": (lambda n_features=None: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64, n_features=n_features)),
    #
    '~b-reg-NSV-64': _noshapevar(lambda n_features=None: _problem_for_predictor_regression(
        dtype=numpy.float64, n_features=n_features)),
    '~m-reg-NSV-64': _noshapevar(lambda n_features=None: _problem_for_predictor_multi_regression(
        dtype=numpy.float64, n_features=n_features)),
    "~b-reg-std-NSV-64": (_noshapevar(lambda n_features=None: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64, n_features=n_features))),
    "~m-reg-std-NSV-64": (_noshapevar(lambda n_features=None: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64, n_features=n_features))),
    # isotonic
    "~b-reg-1d": _1d_problem(_problem_for_predictor_regression),
    '~num+y-tr-1d': _1d_problem(_problem_for_numerical_trainable_transform),
    # text
    "key-int-col": _problem_for_dict_vectorizer,
    "key-str-col": _problem_for_feature_hasher,
    "int-col": _problem_for_label_encoder,
    "one-hot": _problem_for_one_hot_encoder,
    'text-col': _problem_for_tfidf_vectorizer,
    'bow': _problem_for_tfidf_transformer,
}
