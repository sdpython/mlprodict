# pylint: disable=E0611
"""
@brief      test log(time=15s)
"""
import unittest
import os
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        get_all_opkernel_def, get_all_operator_schema)
except (ImportError, AttributeError):
    get_all_opkernel_def = None
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import (
    loadop, OnnxOperatorFunction, _CustomSchema, __file__ as xop_file)


class TestXOpsOrt(ExtTestCase):

    @unittest.skipIf(get_all_opkernel_def is None,
                     reason="onnxruntime not compiled with flag --gen_doc.")
    def test_onnxruntime_serialize(self):
        data = []
        for op in get_all_operator_schema():
            if op.domain in ('', 'ai.onnx.ml', 'ai.onnx.preview.training'):
                continue
            sch = _CustomSchema(op)
            data.append(sch.SerializeToString())

        temp = get_temp_folder(__file__, "temp_get_all_operator_schema")
        ser = os.path.join(temp, "ort_get_all_operator_schema.txt")
        with open(ser, "w") as f:
            f.write("%d\n" % len(data))
            for d in data:
                f.write("%s\n" % d.replace(" ", ""))

        current = os.path.join(os.path.dirname(xop_file),
                               "ort_get_all_operator_schema.txt")
        size1 = os.lstat(ser).st_size
        size2 = os.lstat(current).st_size
        self.assertEqual(size1, size2)

    def test_onnxruntime_inverse(self):
        # See https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md.
        OnnxAbs = loadop(('', "Abs"))
        OnnxInverse = loadop(("com.microsoft", "Inverse"))
        ov = OnnxAbs('X')
        inv = OnnxInverse(ov, output_names=['Y'],
                          domain='com.microsoft')
        onx = inv.to_onnx(numpy.float32, numpy.float32)

        x = numpy.array([[1, 0.5], [0.2, 5]], dtype=numpy.float32)
        i = numpy.linalg.inv(x)
        oinf = OnnxInference(onx, runtime='onnxruntime1')
        got = oinf.run({'X': x})
        self.assertEqualArray(i, got['Y'])

        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({'X': x})
        self.assertEqualArray(i, got['Y'])

    def test_onnxruntime_inverse_nodomain(self):
        # See https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md.
        OnnxAbs = loadop(('', "Abs"))
        OnnxInverse = loadop(("com.microsoft", "Inverse"))
        ov = OnnxAbs('X')
        inv = OnnxInverse(ov, output_names=['Y'])
        onx = inv.to_onnx(numpy.float32, numpy.float32)

        x = numpy.array([[1, 0.5], [0.2, 5]], dtype=numpy.float32)
        i = numpy.linalg.inv(x)
        oinf = OnnxInference(onx, runtime='onnxruntime1')
        got = oinf.run({'X': x})
        self.assertEqualArray(i, got['Y'])

        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({'X': x})
        self.assertEqualArray(i, got['Y'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
