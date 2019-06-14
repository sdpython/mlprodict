# -*- encoding: utf-8 -*-
# pylint: disable=W0611
"""
@file
@brief Imports runtime operators.
"""

from ._op import OpRun
from .op_add import Add
from .op_cast import Cast
from .op_linear_classifier import LinearClassifier
from .op_linear_regressor import LinearRegressor
from .op_normalizer import Normalizer
from .op_zipmap import ZipMap


from ..doc_helper import get_rst_doc
clo = locals().copy()
for name, cl in clo.items():
    if not cl.__doc__ and issubclass(cl, OpRun):
        cl.__doc__ = get_rst_doc(cl.__name__)
