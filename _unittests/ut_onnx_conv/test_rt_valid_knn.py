"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
from onnx.defs import onnx_opset_version
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, skipif_circleci, unittest_require_at_least
import skl2onnx
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets
from mlprodict.onnxrt.validate.validate import RuntimeBadResultsError


class TestRtValidateKNN(ExtTestCase):

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @skipif_circleci("no end")
    def test_rt_knn_regressor(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 2 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        def fil(m, p, s, e, o):
            if o == [{}] or o == {}:  # pylint: disable=R1714
                return False
            if "-64" not in p and 'b-reg' in p:
                return True
            return False

        try:
            rows = list(enumerate_validated_operator_opsets(
                verbose, models={"KNeighborsRegressor"},
                opset_min=onnx_opset_version(),
                opset_max=onnx_opset_version(), fLOG=myprint, fail_bad_results=True,
                runtime='python', debug=debug, extended_list=True,
                store_models=True,
                filter_scenario=lambda m, p, s, e, o: fil(m, p, s, e, o)))
        except RuntimeBadResultsError as e:
            obs = e.obs
            exp = obs['lambda-skl'][0](obs['lambda-skl'][1])
            got = obs['lambda-batch'][0](obs['lambda-batch'][1])
            diff = numpy.abs(exp - got['variable'])
            best = numpy.argmax(diff)
            f1 = obs['lambda-skl'][1][best:best + 1]
            f2 = obs['lambda-batch'][1][best:best + 1]
            exp = obs['lambda-skl'][0](f1)
            got = obs['lambda-batch'][0](f2)
            model = obs['MODEL']
            oinf = obs['OINF']
            got = oinf.run({'X': f2}, verbose=3, fLOG=print)
            exp = model.predict(f1.astype(numpy.float32))
            rows = [str(f1), str(f2), str(got), str(exp)]
            raise RuntimeError("\n".join(rows)) from e
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
