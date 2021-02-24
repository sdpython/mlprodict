"""
@file
@brief Scenario for additional converters.
"""
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier


def find_suitable_problem(model):
    """
    Defines suitables problems for additional converters.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning
        :rst:

        from mlprodict.onnx_conv.validate_scenarios import find_suitable_problem
        from mlprodict.onnxrt.validate.validate_helper import sklearn_operators
        from pyquickhelper.pandashelper import df2rst
        from pandas import DataFrame
        res = sklearn_operators(extended=True)
        res = [_ for _ in res if _['package'] != 'sklearn']
        rows = []
        for model in res:
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

    """
    def _internal(model):
        # Exceptions
        if model in {LGBMRegressor, XGBRegressor}:
            return ['b-reg', '~b-reg-64']

        if model in {LGBMClassifier, XGBClassifier}:
            return ['b-cl', 'm-cl', '~b-cl-64']

        # Not in this list
        return None

    res = _internal(model)
    return res


def build_custom_scenarios():
    """
    Defines parameters values for some operators.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.onnx_conv.validate_scenarios import build_custom_scenarios
        import pprint
        pprint.pprint(build_custom_scenarios())
    """
    return {
        # scenarios
        LGBMClassifier: [
            ('default', {'n_estimators': 5}, {'conv_options': [
             {LGBMClassifier: {'zipmap': False}}]}),
        ],
        LGBMRegressor: [
            ('default', {'n_estimators': 100}),
        ],
        XGBClassifier: [
            ('default', {'n_estimators': 5}, {'conv_options': [
             {XGBClassifier: {'zipmap': False}}]}),
        ],
        XGBRegressor: [
            ('default', {'n_estimators': 100}),
        ],
    }
