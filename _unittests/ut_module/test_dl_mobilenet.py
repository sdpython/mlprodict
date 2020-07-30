"""
@brief      test log(time=7s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from pyensae.datasource import download_data
from mlprodict.onnxrt import OnnxInference


class TestLONGMobileNet(ExtTestCase):

    def test_mobilenet(self):
        src = ("https://s3.amazonaws.com/onnx-model-zoo/mobilenet/"
               "mobilenetv2-1.0/")
        model_file = "mobilenetv2-1.0.onnx"
        download_data(model_file, website=src)
        X = numpy.random.rand(1, 3, 224, 224).astype(dtype=numpy.float32)
        rts = ['python', 'python_compiled_debug',
               'python_compiled', 'onnxruntime1']
        res = []
        for i, rt in enumerate(rts):
            oinf = OnnxInference(model_file, runtime=rt)
            self.assertNotEmpty(oinf)
            self.assertEqual(oinf.input_names[:1], ['data'])
            if hasattr(oinf, 'inits_'):
                self.assertIn(
                    "mobilenetv20_features_conv0_weight", oinf.inits_)
                self.assertEqualArray(
                    (0, -1), oinf.inits_["reshape_attr_tensor421"]['value'])
            name = oinf.input_names[0]
            out = oinf.output_names[0]
            if 'debug' in rt:
                Y, stdout, _ = self.capture(
                    lambda oi=oinf: oi.run({name: X}))  # pylint: disable=W0640
                self.assertIn('-=', stdout)
            else:
                Y = oinf.run({name: X})
            if any(map(numpy.isnan, Y[out].ravel())):
                raise AssertionError(
                    "Runtime {}:{} produces NaN.\n{}".format(i, rt, Y[out]))
            res.append((rt, Y[out]))
        for rt, r in res[1:]:
            exp = numpy.squeeze(r[0])
            got = numpy.squeeze(r)
            try:
                self.assertEqual(exp.shape, got.shape)
                self.assertEqualArray(got, exp)
            except AssertionError as e:
                raise AssertionError(
                    "Issue with runtime: '{}'.".format(rt)) from e


if __name__ == "__main__":
    unittest.main()
