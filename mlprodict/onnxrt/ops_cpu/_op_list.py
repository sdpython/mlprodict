# -*- encoding: utf-8 -*-
# pylint: disable=W0611
"""
@file
@brief Imports runtime operators.
"""

from ._op import OpRun
from .op_add import Add
from .op_argmax import ArgMax
from .op_argmin import ArgMin
from .op_cast import Cast
from .op_div import Div
from .op_gemm import Gemm
from .op_linear_classifier import LinearClassifier
from .op_linear_regressor import LinearRegressor
from .op_mul import Mul
from .op_normalizer import Normalizer
from .op_reduce_sum import ReduceSum
from .op_reduce_sum_square import ReduceSumSquare
from .op_scaler import Scaler
from .op_sqrt import Sqrt
from .op_sub import Sub
from .op_zipmap import ZipMap


from ..doc_helper import get_rst_doc
clo = locals().copy()
for name, cl in clo.items():
    if not cl.__doc__ and issubclass(cl, OpRun):
        cl.__doc__ = get_rst_doc(cl.__name__)
