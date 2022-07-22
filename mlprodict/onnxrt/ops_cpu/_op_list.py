# -*- encoding: utf-8 -*-
# pylint: disable=W0611
"""
@file
@brief Imports runtime operators.
"""

from ._op import OpRun
from .op_abs import Abs
from .op_acos import Acos
from .op_acosh import Acosh
from .op_adagrad import Adagrad
from .op_adam import Adam
from .op_add import Add
from .op_and import And
from .op_argmax import ArgMax
from .op_argmin import ArgMin
from .op_array_feature_extractor import ArrayFeatureExtractor
from .op_asin import Asin
from .op_asinh import Asinh
from .op_atan import Atan
from .op_atanh import Atanh
from .op_average_pool import AveragePool
from .op_batch_normalization import BatchNormalization, BatchNormalization_14
from .op_binarizer import Binarizer
from .op_bitshift import BitShift
from .op_broadcast_gradient_args import BroadcastGradientArgs
from .op_cast import Cast, CastLike
from .op_cdist import CDist
from .op_ceil import Ceil
from .op_celu import Celu
from .op_clip import Clip_6, Clip_11, Clip
from .op_category_mapper import CategoryMapper
from .op_complex_abs import ComplexAbs
from .op_compress import Compress
from .op_concat import Concat
from .op_concat_from_sequence import ConcatFromSequence
from .op_conv import Conv
from .op_conv_transpose import ConvTranspose
from .op_constant import Constant, Constant_12, Constant_11, Constant_9
from .op_constant_of_shape import ConstantOfShape
from .op_cos import Cos
from .op_cosh import Cosh
from .op_cum_sum import CumSum
from .op_debug import DEBUG
from .op_det import Det
from .op_depth_to_space import DepthToSpace, SpaceToDepth
from .op_dequantize_linear import DequantizeLinear
from .op_dict_vectorizer import DictVectorizer
from .op_div import Div
from .op_dropout import Dropout, Dropout_7, Dropout_12
from .op_einsum import Einsum
from .op_elu import Elu
from .op_equal import Equal
from .op_erf import Erf
from .op_exp import Exp
from .op_expand import Expand, Expand_13
from .op_expression import Expression
from .op_eyelike import EyeLike
from .op_feature_vectorizer import FeatureVectorizer
from .op_fft import FFT
from .op_fft2d import FFT2D
from .op_flatten import Flatten
from .op_fused_matmul import FusedMatMul
from .op_gather import Gather
from .op_gathernd import GatherND
from .op_gather_elements import GatherElements
from .op_gemm import Gemm
from .op_global_average_pool import GlobalAveragePool, GlobalMaxPool
from .op_greater import Greater, GreaterOrEqual
from .op_grid_sample import GridSample
from .op_gru import GRU
from .op_hardmax import Hardmax
from .op_hard_sigmoid import HardSigmoid
from .op_floor import Floor
from .op_identity import Identity
from .op_if import If
from .op_imputer import Imputer
from .op_inverse import Inverse
from .op_isinf import IsInf
from .op_isnan import IsNaN
from .op_label_encoder import LabelEncoder
from .op_leaky_relu import LeakyRelu
from .op_less import Less, LessOrEqual
from .op_linear_classifier import LinearClassifier
from .op_linear_regressor import LinearRegressor
from .op_log import Log
from .op_log_softmax import LogSoftmax
from .op_loop import Loop
from .op_lp_normalization import LpNormalization
from .op_lrn import LRN
from .op_lstm import LSTM
from .op_matmul import MatMul
from .op_max import Max
from .op_max_pool import MaxPool
from .op_mean import Mean
from .op_min import Min
from .op_mod import Mod
from .op_momentum import Momentum
from .op_mul import Mul
from .op_neg import Neg
from .op_negative_log_likelihood_loss import NegativeLogLikelihoodLoss
from .op_normalizer import Normalizer
from .op_non_max_suppression import NonMaxSuppression
from .op_non_zero import NonZero
from .op_not import Not
from .op_one_hot import OneHot
from .op_one_hot_encoder import OneHotEncoder
from .op_or import Or
from .op_pad import Pad
from .op_pow import Pow
from .op_prelu import PRelu
from .op_quantize_linear import QuantizeLinear, DynamicQuantizeLinear
from .op_qlinear_conv import QLinearConv
from .op_random import (
    Bernoulli, RandomNormal, RandomUniform,
    RandomUniformLike, RandomNormalLike)
from .op_range import Range
from .op_reciprocal import Reciprocal
from .op_reduce_log_sum import ReduceLogSum
from .op_reduce_log_sum_exp import ReduceLogSumExp
from .op_reduce_l1 import ReduceL1
from .op_reduce_l2 import ReduceL2
from .op_reduce_min import ReduceMin
from .op_reduce_max import ReduceMax
from .op_reduce_mean import ReduceMean
from .op_reduce_prod import ReduceProd
from .op_reduce_sum import (
    ReduceSum_1, ReduceSum_11, ReduceSum_13, ReduceSum)
from .op_reduce_sum_square import ReduceSumSquare
from .op_relu import Relu, ThresholdedRelu
from .op_reshape import Reshape, Reshape_5, Reshape_13, Reshape_14
from .op_resize import Resize
from .op_rfft import RFFT
from .op_roi_align import RoiAlign
from .op_round import Round
from .op_rnn import RNN
from .op_scaler import Scaler
from .op_scan import Scan
from .op_scatter_elements import ScatterElements
from .op_scatternd import ScatterND
from .op_softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss
from .op_selu import Selu
from .op_sequence_at import SequenceAt
from .op_sequence_construct import SequenceConstruct
from .op_sequence_empty import SequenceEmpty
from .op_sequence_insert import SequenceInsert
from .op_shape import Shape
from .op_shrink import Shrink
from .op_sigmoid import Sigmoid
from .op_sign import Sign
from .op_sin import Sin
from .op_sinh import Sinh
from .op_size import Size
from .op_slice import Slice, Slice_1, Slice_10
from .op_split import Split, Split_2, Split_11, Split_13
from .op_softmax import Softmax, SoftmaxGrad, SoftmaxGrad_13
from .op_softplus import Softplus
from .op_softsign import Softsign
from .op_solve import Solve
from .op_sqrt import Sqrt
from .op_squeeze import Squeeze, Squeeze_1, Squeeze_11, Squeeze_13
from .op_string_normalizer import StringNormalizer
from .op_sub import Sub
from .op_sum import Sum
from .op_svm_classifier import SVMClassifier, SVMClassifierDouble
from .op_svm_regressor import SVMRegressor, SVMRegressorDouble
from .op_tan import Tan
from .op_tanh import Tanh
from .op_tfidfvectorizer import TfIdfVectorizer
from .op_tokenizer import Tokenizer
from .op_topk import TopK_10, TopK_11, TopK_1, TopK
from .op_transpose import Transpose
from .op_tree_ensemble_classifier import (
    TreeEnsembleClassifierDouble,
    TreeEnsembleClassifier_1, TreeEnsembleClassifier_3, TreeEnsembleClassifier)
from .op_tree_ensemble_regressor import (
    TreeEnsembleRegressorDouble,
    TreeEnsembleRegressor_1, TreeEnsembleRegressor_3, TreeEnsembleRegressor)
from .op_trilu import Trilu
from .op_unique import Unique
from .op_unsqueeze import Unsqueeze, Unsqueeze_1, Unsqueeze_11, Unsqueeze_13
from .op_where import Where
from .op_xor import Xor
from .op_yield_op import YieldOp
from .op_zipmap import ZipMap


from ..doc.doc_helper import get_rst_doc
_op_list = []
clo = locals().copy()
for name, cl in clo.items():
    if "_" in name:
        continue
    if name in {'cl', 'clo', 'name'}:
        continue  # pragma: no cover
    if not cl.__doc__ and issubclass(cl, OpRun):
        cl.__doc__ = get_rst_doc(cl.__name__)
        _op_list.append(cl)
