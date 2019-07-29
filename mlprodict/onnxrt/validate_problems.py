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
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
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
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC, LinearSVR, NuSVR, SVR, SVC, NuSVC
from sklearn.utils import shuffle
from skl2onnx.common.data_types import (
    FloatTensorType, DoubleTensorType, StringTensorType, DictionaryType
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
    m-cl classification problem.
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
    m-cl classification problem.
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
    mregression problem.
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


def _problem_for_numerical_trainable_transform(dtype=numpy.float32):
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


def _problem_for_cl_decision_function(dtype=numpy.float32):
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y, [('X', X[:1].astype(dtype))],
            'decision_function', 0, X.astype(dtype))


def _problem_for_cl_decision_function_binary(dtype=numpy.float32):
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
            'decision_function', 0, X.astype(dtype))


def _problem_for_label_encoder(dtype=numpy.int64):
    """
    Returns a problem for the :epkg:`sklearn:preprocessing:LabelEncoder`.
    """
    data = load_iris()
    # X = data.data
    y = data.target.astype(dtype)
    itt = [('X', y[:1].astype(dtype))]
    return (y, None, itt, 'transform', 0, y)


def _problem_for_dict_vectorizer(dtype=numpy.float32):
    """
    Returns a problem for the :epkg:`sklearn:feature_extraction:DictVectorizer`.
    """
    data = load_iris()
    # X = data.data
    y = data.target
    y2 = [{_: dtype(1000 + i)} for i, _ in enumerate(y)]
    y2[0][2] = -2
    itt = [("X", DictionaryType(StringTensorType([1]), FloatTensorType([1])))]
    y2 = numpy.array(y2)
    return (y2, y, itt, 'transform', 0, y2)


def _problem_for_feature_hasher(dtype=numpy.float32):
    """
    Returns a problem for the :epkg:`sklearn:feature_extraction:DictVectorizer`.
    """
    data = load_iris()
    # X = data.data
    y = data.target
    y2 = [{("cl%d" % _): dtype(1000 + i)} for i, _ in enumerate(y)]
    y2[0]["cl2"] = -2
    itt = [("X", DictionaryType(StringTensorType([1]), FloatTensorType([1])))]
    y2 = numpy.array(y2)
    return (y2, y, itt, 'transform', 0, y2)


def _problem_for_one_hot_encoder(dtype=numpy.float32):
    """
    Returns a problem for the :epkg:`sklearn:preprocessing:OneHotEncoder`.
    """
    data = load_iris()
    X = data.data.astype(numpy.int32).astype(dtype)
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
    * `scoring`: transform numerical features, target is usually needed
    * `outlier`: outlier prediction
    * `linearsvc`: classifier without *predict_proba*
    * `cluster`: similar to transform
    * `num+y-tr`: similar to transform with targets
    * `num-tr-clu`: similar to cluster, but returns
        scores or distances instead of cluster
    * `key-col`: list of dictionaries

    Suffix `nofit` indicates the predictions happens
    without the model being fitted. This is the case
    for :epkg:`sklearn:gaussian_process:GaussianProcessRegressor`.
    The suffix `-cov` indicates the method `predict` was called
    with parameter ``return_cov=True``, `-std` tells
    method `predict` was called with parameter ``return_std=True``.
    The suffix ``-NSV`` creates an input variable
    like the following ``[('X', FloatTensorType(["da", "db"]))]``.
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
        :rst:

        from mlprodict.onnxrt.validate import sklearn_operators, find_suitable_problem
        from pyquickhelper.pandashelper import df2rst
        from pandas import DataFrame
        res = sklearn_operators()
        rows = []
        for model in res[:20]:
            name = model['name']
            row = dict(name=name)
            try:
                prob = find_suitable_problem(model['cl'])
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
    def _internal(model):
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

        if model in {FeatureHasher}:
            return ['key-str-col']

        if model in {OneHotEncoder}:
            return ['one-hot']

        if model in {LabelBinarizer, LabelEncoder}:
            return ['int-col']

        if model in {BaggingClassifier, BernoulliNB, CalibratedClassifierCV,
                     ComplementNB, GaussianNB, GaussianProcessClassifier,
                     GradientBoostingClassifier, LabelPropagation, LabelSpreading,
                     LinearDiscriminantAnalysis, LogisticRegressionCV,
                     MultinomialNB, NuSVC, QuadraticDiscriminantAnalysis,
                     RandomizedSearchCV, SGDClassifier, SVC}:
            return ['b-cl', 'm-cl']

        if model in {Perceptron}:
            return ['~b-cl-nop', '~m-cl-nop', '~b-cl-dec', '~m-cl-dec']

        if model in {AdaBoostRegressor}:
            return ['b-reg']

        if model in {LinearSVC, NearestCentroid}:
            return ['~b-cl-nop', '~b-cl-nop-64']

        if model in {RFE, RFECV}:
            return ['b-cl', 'm-cl', 'b-reg']

        if model in {GridSearchCV}:
            return ['b-cl', 'm-cl',
                    'b-reg', 'm-reg',
                    '~b-reg-64', '~b-cl-64',
                    'cluster', 'outlier', '~m-label']

        if model in {VotingClassifier}:
            return ['b-cl']

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
                     PassiveAggressiveClassifier, RadiusNeighborsClassifier,
                     RidgeClassifier, RidgeClassifierCV}:
            return ['~b-cl-nop', '~m-cl-nop']

        # trainable transform
        if model in {GenericUnivariateSelect,
                     NeighborhoodComponentsAnalysis,
                     PLSSVD, SelectFwe, SelectKBest,
                     SelectPercentile}:
            return ["num+y-tr"]

        # no m-label
        if model in {AdaBoostClassifier, LogisticRegression}:
            return ['b-cl', '~b-cl-64', 'm-cl']

        # predict, predict_proba
        if hasattr(model, 'predict_proba'):
            if model is OneVsRestClassifier:
                return ['m-cl', '~m-label']
            else:
                return ['b-cl', 'm-cl', '~m-label']

        if hasattr(model, 'predict'):
            if "Classifier" in str(model):
                return ['b-cl', '~b-cl-64', 'm-cl', '~m-label']
            elif "Regressor" in str(model):
                return ['b-reg', 'm-reg', '~b-reg-64']

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
            res.extend(['b-cl', '~b-cl-64', 'm-cl', '~m-label'])
        if issubclass(model, RegressorMixin):
            res.extend(['b-reg', 'm-reg', '~b-reg-64', '~m-reg-64'])

        if len(res) == 0 and hasattr(model, 'fit') and hasattr(model, 'score'):
            return ['~scoring']
        if len(res) > 0:
            return res

        raise RuntimeError("Unable to find problem for model '{}' - {}."
                           "".format(model.__name__, model.__bases__))

    res = _internal(model)
    for r in res:
        if r not in _problems:
            raise ValueError("Unrecognized problem '{}' in\n{}".format(
                r, "\n".join(sorted(_problems))))
    return res


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
            # name = "V%s_" % str(id(Xort))[:5]
            new_itt = []
            for a, b in itt:
                # shape = [name + str(i) for s in b.shape]
                shape = [None for s in b.shape]
                new_itt.append((a, _guess_noshape(b, shape)))
            return new_itt

    def new_fct(**kwargs):
        X, y, itt, meth, mo, Xort = fct(**kwargs)
        new_itt = process_itt(itt, Xort)
        return X, y, new_itt, meth, mo, Xort
    return new_fct


def _1d_problem(fct):

    def new_fct(**kwargs):
        X, y, itt, meth, mo, Xort = fct(**kwargs)
        new_itt = itt  # process_itt(itt, Xort)
        X = X[:, 0]
        return X, y, new_itt, meth, mo, Xort
    return new_fct


_problems = {
    # standard
    "b-cl": _problem_for_predictor_binary_classification,
    "m-cl": _problem_for_predictor_multi_classification,
    "b-reg": _problem_for_predictor_regression,
    "m-reg": _problem_for_predictor_multi_regression,
    "num-tr": _problem_for_numerical_transform,
    'outlier': _problem_for_outlier,
    'cluster': _problem_for_clustering,
    'num+y-tr': _problem_for_numerical_trainable_transform,
    # others
    '~num-tr-clu': _problem_for_clustering_scores,
    "~m-label": _problem_for_predictor_multi_classification_label,
    "~scoring": _problem_for_numerical_scoring,
    '~b-cl-nop': _problem_for_clnoproba_binary,
    '~m-cl-nop': _problem_for_clnoproba,
    '~b-cl-dec': _problem_for_cl_decision_function_binary,
    '~m-cl-dec': _problem_for_cl_decision_function,
    # 64
    "~b-cl-64": lambda: _problem_for_predictor_binary_classification(dtype=numpy.float64),
    "~b-reg-64": lambda: _problem_for_predictor_regression(dtype=numpy.float64),
    '~b-cl-nop-64': lambda: _problem_for_clnoproba(dtype=numpy.float64),
    '~b-clu-64': lambda: _problem_for_clustering(dtype=numpy.float64),
    '~b-cl-dec-64': lambda: _problem_for_cl_decision_function_binary(dtype=numpy.float64),
    '~num-tr-clu-64': lambda: _problem_for_clustering_scores(dtype=numpy.float64),
    "~m-reg-64": lambda: _problem_for_predictor_multi_regression(dtype=numpy.float64),
    #
    "~b-cl-NF": (lambda: _problem_for_predictor_binary_classification() + (False, )),
    "~m-cl-NF": (lambda: _problem_for_predictor_multi_classification() + (False, )),
    "~b-reg-NF": (lambda: _problem_for_predictor_regression() + (False, )),
    "~m-reg-NF": (lambda: _problem_for_predictor_multi_regression() + (False, )),
    #
    "~b-cl-NF-64": (lambda: _problem_for_predictor_binary_classification(dtype=numpy.float64) + (False, )),
    "~m-cl-NF-64": (lambda: _problem_for_predictor_multi_classification(dtype=numpy.float64) + (False, )),
    "~b-reg-NF-64": (lambda: _problem_for_predictor_regression(dtype=numpy.float64) + (False, )),
    "~m-reg-NF-64": (lambda: _problem_for_predictor_multi_regression(dtype=numpy.float64) + (False, )),
    # GaussianProcess
    "~b-reg-NF-cov-64": (lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}},
        return_cov=True, dtype=numpy.float64) + (False, )),
    "~m-reg-NF-cov-64": (lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}},
        return_cov=True, dtype=numpy.float64) + (False, )),
    #
    "~b-reg-NF-std-64": (lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64) + (False, )),
    "~m-reg-NF-std-64": (lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64) + (False, )),
    #
    "~b-reg-cov-64": (lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}},
        return_cov=True, dtype=numpy.float64)),
    "~m-reg-cov-64": (lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_cov": True}},
        return_cov=True, dtype=numpy.float64)),
    #
    "~reg-std-64": (lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64)),
    "~m-reg-std-64": (lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64)),
    #
    '~b-reg-NSV-64': _noshapevar(lambda: _problem_for_predictor_regression(dtype=numpy.float64)),
    '~m-reg-NSV-64': _noshapevar(lambda: _problem_for_predictor_multi_regression(dtype=numpy.float64)),
    "~b-reg-std-NSV-64": (_noshapevar(lambda: _problem_for_predictor_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64))),
    "~m-reg-std-NSV-64": (_noshapevar(lambda: _problem_for_predictor_multi_regression(
        True, options={GaussianProcessRegressor: {"return_std": True}},
        return_std=True, dtype=numpy.float64))),
    # isotonic
    "~b-reg-1d": _1d_problem(_problem_for_predictor_regression),
    '~num+y-tr-1d': _1d_problem(_problem_for_numerical_trainable_transform),
    # text
    "key-int-col": _problem_for_dict_vectorizer,
    "key-str-col": _problem_for_feature_hasher,
    "int-col": _problem_for_label_encoder,
    "one-hot": _problem_for_one_hot_encoder,
}
