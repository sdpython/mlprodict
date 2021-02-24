"""
@file
@brief Datasets to tests models.
"""
import os
from pandas import read_csv


def load_audit():
    """
    Use to test conversion of
    :epkg:`sklearn:ensemble:GradientBoostingClassifier`
    into :epkg:`ONNX`.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.onnxrt.validate.data import load_audit
        df = load_audit()
        print(df.head())
    """
    name = os.path.dirname(__file__)
    name = os.path.join(name, 'audit.csv')
    df = read_csv(name).drop(['ID', 'index'], axis=1, inplace=False).dropna()
    return df
