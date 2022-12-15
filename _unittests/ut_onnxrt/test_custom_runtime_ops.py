"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from numpy.linalg import eig, eigvals
from onnx import TensorProto  # pylint: disable=W0611
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from sklearn.base import TransformerMixin, BaseEstimator
from skl2onnx.algebra import OnnxOperator
from skl2onnx.common.data_types import (
    guess_numpy_type, guess_proto_type)
from skl2onnx import to_onnx
from skl2onnx import update_registered_converter
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd,
    OnnxCast, OnnxCastLike,
    OnnxDiv,
    OnnxGatherElements,
    OnnxEyeLike,
    OnnxMatMul,
    OnnxMul,
    OnnxPow,
    OnnxReduceMean_13,
    OnnxShape,
    OnnxSub,
    OnnxTranspose,
)
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.ops_cpu import OpRunCustom, register_operator


class LiveDecorrelateTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, alpha=0.):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.alpha = alpha

    def fit(self, X, y=None, sample_weights=None):
        if sample_weights is not None:
            raise NotImplementedError(
                "sample_weights != None is not implemented.")
        self.nf_ = X.shape[1]  # pylint: disable=W0201
        return self

    def transform(self, X):
        mean_ = numpy.mean(X, axis=0, keepdims=True)
        X2 = X - mean_
        V = X2.T @ X2 / X2.shape[0]
        if self.alpha != 0:
            V += numpy.identity(V.shape[0]) * self.alpha
        L, P = numpy.linalg.eig(V)
        Linv = L ** (-0.5)
        diag = numpy.diag(Linv)
        root = P @ diag @ P.transpose()
        coef_ = root
        return (X - mean_) @ coef_


class OnnxEig(OnnxOperator):
    """
    Defines a custom operator not defined by ONNX
    specifications but in onnxruntime.
    """

    since_version = 1
    expected_inputs = [('X', 'T')]
    expected_outputs = ['EigenValues', 'EigenVectors']
    input_range = [1, 1]
    output_range = [1, 2]
    is_deprecated = False
    domain = 'onnxcustom'
    operator_name = 'Eig'
    past_version = {}

    def __init__(self, X, eigv=False, op_version=None, **kwargs):
        """
        :param X: array or OnnxOperatorMixin
        :param eigv: also produces the eigen vectors
        :param op_version: opset version
        :param kwargs: additional parameters
        """
        OnnxOperator.__init__(
            self, X, eigv=eigv, op_version=op_version, **kwargs)


def live_decorrelate_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    output_type = input_type([input_dim, op.nf_])
    operator.outputs[0].type = output_type


def live_decorrelate_transformer_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    # We retrieve the unique input.
    X = operator.inputs[0]
    proto_dtype = guess_proto_type(X.type)

    dtype = guess_numpy_type(X.type)

    # new part

    # mean_ = numpy.mean(X, axis=0, keepdims=True)
    mean = OnnxReduceMean_13(X, axes=[0], keepdims=1, op_version=opv)
    mean.set_onnx_name_prefix('mean')

    # X2 = X - mean_
    X2 = OnnxSub(X, mean, op_version=opv)

    # V = X2.T @ X2 / X2.shape[0]
    N = OnnxGatherElements(
        OnnxShape(X, op_version=opv),
        numpy.array([0], dtype=numpy.int64),
        op_version=opv)
    Nf = OnnxCast(N, to=proto_dtype, op_version=opv)
    Nf.set_onnx_name_prefix('N')

    V = OnnxDiv(
        OnnxMatMul(OnnxTranspose(X2, op_version=opv),
                   X2, op_version=opv),
        Nf, op_version=opv)
    V.set_onnx_name_prefix('V1')

    # V += numpy.identity(V.shape[0]) * self.alpha
    V = OnnxAdd(V,
                op.alpha * numpy.identity(op.nf_, dtype=dtype),
                op_version=opv)
    V.set_onnx_name_prefix('V2')

    # L, P = numpy.linalg.eig(V)
    LP = OnnxEig(V, eigv=True, op_version=opv)
    LP.set_onnx_name_prefix('LP')

    # Linv = L ** (-0.5)
    Linv = OnnxPow(LP[0], numpy.array([-0.5], dtype=dtype),
                   op_version=opv)
    Linv.set_onnx_name_prefix('Linv')

    # diag = numpy.diag(Linv)
    diag = OnnxMul(
        OnnxCastLike(OnnxEyeLike(Linv, k=0, op_version=opv), V,
                     op_version=opv),
        Linv, op_version=opv)
    diag.set_onnx_name_prefix('diag')

    # root = P @ diag @ P.transpose()
    trv = OnnxTranspose(LP[1], op_version=opv)
    coef_left = OnnxMatMul(LP[1], diag, op_version=opv)
    coef_left.set_onnx_name_prefix('coef_left')
    coef = OnnxMatMul(coef_left, trv, op_version=opv)
    coef.set_onnx_name_prefix('coef')

    # Same part as before.
    Y = OnnxMatMul(X2, coef, op_version=opv, output_names=out[:1])
    Y.set_onnx_name_prefix('Y')
    Y.add_to(scope, container)


class OpEig(OpRunCustom):  # pylint: disable=W0223

    op_name = 'Eig'
    atts = {'eigv': True}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunCustom.__init__(self, onnx_node, desc=desc,
                             expected_attributes=OpEig.atts,
                             **options)

    def run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.eigv:  # pylint: disable=E1101
            return eig(x)
        return (eigvals(x), )


class TestCustomRuntimeOps(ExtTestCase):

    def test_custom_runtome_ops(self):

        update_registered_converter(
            LiveDecorrelateTransformer, "SklearnLiveDecorrelateTransformer",
            live_decorrelate_transformer_shape_calculator,
            live_decorrelate_transformer_converter)

        data = load_iris()
        X = data.data
        dec = LiveDecorrelateTransformer()
        dec.fit(X)

        onx = to_onnx(dec, X.astype(numpy.float64),
                      target_opset=17)
        self.assertRaise(lambda: OnnxInference(onx), RuntimeError)

        register_operator(OpEig, name='Eig', overwrite=False)
        exp = dec.transform(X.astype(numpy.float32))

        for rt in ['python']:
            with self.subTest(runtime=rt):
                oinf = OnnxInference(onx, runtime=rt)
                got = oinf.run({'X': X.astype(numpy.float64)})
                self.assertEqualArray(exp, got['variable'], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
