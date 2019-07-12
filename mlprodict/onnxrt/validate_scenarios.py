"""
@file
@brief Scenarios for validation.
"""
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from sklearn.decomposition import SparseCoder
from sklearn.ensemble import VotingClassifier, AdaBoostRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, DotProduct, RationalQuadratic
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier, ClassifierChain, RegressorChain
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeRegressor


def build_custom_scenarios():
    """
    Defines parameters values for some operators.

    .. runpython::
        :showcode:

        from mlprodict.onnxrt.validate_scenarios import build_custom_scenarios
        import pprint
        pprint.pprint(build_custom_scenarios())
    """
    return {
        # skips
        SparseCoder: None,
        # scenarios
        AdaBoostRegressor: [
            ('default', {
                'n_estimators': 5,
            }),
        ],
        ClassifierChain: [
            ('logreg', {
                'base_estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        GaussianProcessRegressor: [
            ('expsine', {
                'kernel': ExpSineSquared(),
                'alpha': 20.,
            }),
            ('dotproduct', {
                'kernel': DotProduct(),
                'alpha': 100.,
            }),
            ('rational', {
                'kernel': RationalQuadratic(),
                'alpha': 100.,
            }),
            ('default', {
                'kernel': None,
                'alpha': 100.,
            }),
        ],
        GridSearchCV: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
                'param_grid': {'fit_intercept': [False, True]},
            }),
            ('reg', {
                'estimator': LinearRegression(),
                'param_grid': {'fit_intercept': [False, True]},
            }),
        ],
        LocalOutlierFactor: [
            ('novelty', {
                'novelty': True,
            }),
        ],
        LogisticRegression: [
            ('liblinear', {
                'solver': 'liblinear',
            }),
        ],
        MultiOutputClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        MultiOutputRegressor: [
            ('linreg', {
                'estimator': LinearRegression(),
            })
        ],
        Normalizer: [
            ('l2', {'norm': 'l2', }),
            ('l1', {'norm': 'l1', }),
            ('max', {'norm': 'max', }),
        ],
        NuSVC: [
            ('prob', {
                'probability': True,
            }),
        ],
        OneVsOneClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        OneVsRestClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        OutputCodeClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        RandomizedSearchCV: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
                'param_distributions': {'fit_intercept': [False, True]},
            }),
            ('reg', {
                'estimator': LinearRegression(),
                'param_distributions': {'fit_intercept': [False, True]},
            }),
        ],
        RegressorChain: [
            ('linreg', {
                'base_estimator': LinearRegression(),
            })
        ],
        RFE: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
            }),
            ('reg', {
                'estimator': LinearRegression(),
            })
        ],
        RFECV: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
            }),
            ('reg', {
                'estimator': LinearRegression(),
            })
        ],
        SelectFromModel: [
            ('rf', {
                'estimator': DecisionTreeRegressor(),
            }),
        ],
        SGDClassifier: [
            ('log', {
                'loss': 'log',
            }),
        ],
        SVC: [
            ('prob', {
                'probability': True,
            }),
        ],
        VotingClassifier: [
            ('logreg-noflatten', {
                'voting': 'soft',
                'flatten_transform': False,
                'estimators': [
                    ('lr1', LogisticRegression(solver='liblinear')),
                    ('lr2', LogisticRegression(
                        solver='liblinear', fit_intercept=False)),
                ],
            })
        ],
        VotingRegressor: [
            ('linreg', {
                'estimators': [
                    ('lr1', LinearRegression()),
                    ('lr2', LinearRegression(fit_intercept=False)),
                ],
            })
        ],
    }


_extra_parameters = build_custom_scenarios()
