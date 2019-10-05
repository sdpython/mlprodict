"""
@brief      test log(time=2s)
"""
from collections import OrderedDict
from io import StringIO
import unittest
from logging import getLogger
import numpy
import pandas
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK, Sum
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
from pyquickhelper.texthelper.version_helper import compare_module_version
from onnxruntime import __version__ as ort_version
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
try:
    from skl2onnx.operator_converters.gaussian_process import convert_kernel
except ImportError:
    convert_kernel = None
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate.side_by_side import side_by_side_by_values


Xtest_ = pandas.read_csv(StringIO("""
1.000000000000000000e+02,1.061277971307766705e+02,1.472195004809226493e+00,2.307125069497626552e-02,4.539948095743629591e-02,2.855191098141335870e-01
1.000000000000000000e+02,9.417031896832908444e+01,1.249743892709246573e+00,2.370416174339620707e-02,2.613847280316268853e-02,5.097165413593484073e-01
1.000000000000000000e+02,9.305231488674536422e+01,1.795726729335217264e+00,2.473274733802270642e-02,1.349765645107412620e-02,9.410288840541443378e-02
1.000000000000000000e+02,7.411264142156210255e+01,1.747723020195752319e+00,1.559695663417645997e-02,4.230394035515055301e-02,2.225492746314280956e-01
1.000000000000000000e+02,9.326006195761877393e+01,1.738860294343326229e+00,2.280160135767652502e-02,4.883335335161764074e-02,2.806808409247734115e-01
1.000000000000000000e+02,8.341529291866362428e+01,5.119682123742423929e-01,2.488795768635816003e-02,4.887573336092913834e-02,1.673462179673477768e-01
1.000000000000000000e+02,1.182436477919874562e+02,1.733516391831658954e+00,1.533520930349476820e-02,3.131213519485807895e-02,1.955345358785769427e-01
1.000000000000000000e+02,1.228982583299257101e+02,1.115599996405831629e+00,1.929354155079938959e-02,3.056996308544096715e-03,1.197052763998271013e-01
1.000000000000000000e+02,1.160303269386108838e+02,1.018627021014927303e+00,2.248784981616459844e-02,2.688111547114307651e-02,3.326105131778724355e-01
1.000000000000000000e+02,1.163414374640396005e+02,6.644299545804077667e-01,1.508088417713602906e-02,4.451836657613789106e-02,3.245643044204808425e-01
""".strip("\n\r ")), header=None).values


threshold = "0.4.0"


class TestOnnxrtSideBySide(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @unittest.skipIf(convert_kernel is None, reason="not enough recent version")
    def test_kernel_ker12_def(self):
        ker = (Sum(CK(0.1, (1e-3, 1e3)), CK(0.1, (1e-3, 1e3)) *
                   RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))))
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=numpy.float32,
                             op_version=10)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            outputs=[('Y', FloatTensorType([None, None]))])
        sess = OnnxInference(model_onnx.SerializeToString())
        res = sess.run({'X': Xtest_.astype(numpy.float32)})
        m1 = res['Y']
        m2 = ker(Xtest_)
        self.assertEqualArray(m1, m2)

    @unittest.skipIf(convert_kernel is None, reason="not enough recent version")
    def test_kernel_ker2_def(self):
        ker = Sum(
            CK(0.1, (1e-3, 1e3)) * RBF(length_scale=10,
                                       length_scale_bounds=(1e-3, 1e3)),
            CK(0.1, (1e-3, 1e3)) * RBF(length_scale=1,
                                       length_scale_bounds=(1e-3, 1e3))
        )
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=numpy.float32,
                             op_version=10)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            outputs=[('Y', FloatTensorType([None, None]))])
        sess = OnnxInference(model_onnx.SerializeToString())

        res = sess.run({'X': Xtest_.astype(numpy.float32)})
        m1 = res['Y']
        m2 = ker(Xtest_)
        self.assertEqualArray(m1, m2)

        res = sess.run({'X': Xtest_.astype(numpy.float32)}, intermediate=True)
        self.assertGreater(len(res), 30)
        self.assertIsInstance(res, OrderedDict)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @unittest.skipIf(convert_kernel is None, reason="not enough recent version")
    @unittest.skipIf(compare_module_version(ort_version, threshold) <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_kernel_ker2_def_ort(self):
        ker = Sum(
            CK(0.1, (1e-3, 1e3)) * RBF(length_scale=10,
                                       length_scale_bounds=(1e-3, 1e3)),
            CK(0.1, (1e-3, 1e3)) * RBF(length_scale=1,
                                       length_scale_bounds=(1e-3, 1e3))
        )
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=numpy.float32,
                             op_version=10)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            outputs=[('Y', FloatTensorType([None, None]))])
        sess = OnnxInference(model_onnx.SerializeToString(),
                             runtime="onnxruntime2")
        res = sess.run({'X': Xtest_.astype(numpy.float32)})
        m1 = res['Y']
        m2 = ker(Xtest_)
        self.assertEqualArray(m1, m2, decimal=5)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @unittest.skipIf(convert_kernel is None, reason="not enough recent version")
    @unittest.skipIf(compare_module_version(ort_version, threshold) <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_kernel_ker2_def_ort1(self):
        ker = Sum(
            CK(0.1, (1e-3, 1e3)) * RBF(length_scale=10,
                                       length_scale_bounds=(1e-3, 1e3)),
            CK(0.1, (1e-3, 1e3)) * RBF(length_scale=1,
                                       length_scale_bounds=(1e-3, 1e3))
        )
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=numpy.float32,
                             op_version=10)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            outputs=[('Y', FloatTensorType([None, None]))])
        sess = OnnxInference(model_onnx.SerializeToString(),
                             runtime="onnxruntime1")

        rows = []

        def myprint(*args, **kwargs):
            rows.append(" ".join(map(str, args)))

        res = sess.run({'X': Xtest_.astype(numpy.float32)},
                       intermediate=True, verbose=1, fLOG=myprint)
        self.assertGreater(len(rows), 2)
        m1 = res['Y']
        self.assertNotEmpty(m1)
        self.assertGreater(len(res), 2)
        # m2 = ker(Xtest_)
        # self.assertEqualArray(m1, m2, decimal=5)

        cpu = OnnxInference(model_onnx.SerializeToString())
        sbs = side_by_side_by_values(
            [cpu, sess], inputs={'X': Xtest_.astype(numpy.float32)})
        self.assertGreater(len(sbs), 2)
        self.assertIsInstance(sbs, list)
        self.assertIsInstance(sbs[0], dict)
        self.assertIn('step', sbs[0])
        self.assertIn('step', sbs[1])
        self.assertIn('metric', sbs[0])
        self.assertIn('metric', sbs[1])
        self.assertIn('cmp', sbs[0])
        self.assertIn('cmp', sbs[1])

        sess3 = OnnxInference(model_onnx.SerializeToString(),
                              runtime="onnxruntime2")
        sbs = side_by_side_by_values(
            [cpu, sess, sess3], inputs={'X': Xtest_.astype(numpy.float32)})
        self.assertNotEmpty(sbs)

        inputs = {'X': Xtest_.astype(numpy.float32)}
        sbs = side_by_side_by_values(
            [(cpu, inputs), (sess, inputs), (sess3, inputs)])
        self.assertNotEmpty(sbs)


if __name__ == "__main__":
    unittest.main()
