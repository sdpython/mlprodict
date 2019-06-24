# -*- encoding: utf-8 -*-
# pylint: disable=W0611
"""
@file
@brief Imports runtime operators.
"""

from ._op import OpRun
from .op_abs import Abs
from .op_add import Add
from .op_argmax import ArgMax
from .op_argmin import ArgMin
from .op_array_feature_extractor import ArrayFeatureExtractor
from .op_cast import Cast
from .op_ceil import Ceil
from .op_clip import Clip
from .op_concat import Concat
from .op_div import Div
from .op_exp import Exp
from .op_gemm import Gemm
from .op_greater import Greater
from .op_floor import Floor
from .op_identity import Identity
from .op_linear_classifier import LinearClassifier
from .op_linear_regressor import LinearRegressor
from .op_log import Log
from .op_matmul import MatMul
from .op_mean import Mean
from .op_mul import Mul
from .op_normalizer import Normalizer
from .op_pow import Pow
from .op_reciprocal import Reciprocal
from .op_reduce_log_sum_exp import ReduceLogSumExp
from .op_reduce_prod import ReduceProd
from .op_reduce_sum import ReduceSum
from .op_reduce_sum_square import ReduceSumSquare
from .op_reshape import Reshape
from .op_scaler import Scaler
from .op_slice import Slice
from .op_sqrt import Sqrt
from .op_sub import Sub
from .op_tree_ensemble_classifier import TreeEnsembleClassifier
from .op_tree_ensemble_regressor import TreeEnsembleRegressor
from .op_zipmap import ZipMap


from ..doc_helper import get_rst_doc
_op_list = []
clo = locals().copy()
for name, cl in clo.items():
    if not cl.__doc__ and issubclass(cl, OpRun):
        cl.__doc__ = get_rst_doc(cl.__name__)
        _op_list.append(cl)
