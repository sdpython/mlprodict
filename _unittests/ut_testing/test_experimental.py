"""
@brief      test log(time=6s)
"""
import unittest
import numpy
from onnx import helper, TensorProto
from pyquickhelper.pycode import ExtTestCase, is_travis_or_appveyor
from mlprodict.testing.experimental import custom_pad, custom_einsum
from mlprodict.testing.experimental_c_impl.experimental_c import (  # pylint: disable=E0611,E0401
    custom_einsum_double, custom_einsum_int64, custom_einsum_float,
    code_optimisation, custom_reducesum_rk_double,
    custom_reducesum_rk_float, benchmark_cache, benchmark_cache_tree)
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.tools.ort_wrapper import InferenceSession


class TestExperimental(ExtTestCase):

    def ort_path_pad(self, x, pads):
        pads = list(pads[:, 0]) + list(pads[:, 1])
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, x.shape)  # pylint: disable=E1101
        P = helper.make_tensor_value_info(
            'P', TensorProto.INT64, [len(pads), ])  # pylint: disable=E1101
        Y = helper.make_tensor_value_info(
            'Y', TensorProto.FLOAT, tuple(-1 for s in x.shape))  # pylint: disable=E1101
        npads = numpy.array(pads, dtype=numpy.int64)
        op = helper.make_node('Pad', ['X', 'P'], ['Y'])
        graph = helper.make_graph([op], 'graph', [X, P], [Y])
        model = helper.make_model(
            graph, producer_name='mlprodict', ir_version=6, producer_version='0.1')
        op_set = model.opset_import[0]  # pylint: disable=E1101
        op_set.version = TARGET_OPSET
        sess = InferenceSession(model.SerializeToString())
        return numpy.squeeze(sess.run(['Y'], {'X': x, 'P': npads}))

    def fct_test(self, custom_fct, fct, *inputs, verbose=True):
        got = custom_fct(*inputs, verbose=verbose)
        exp = fct(*inputs)
        try:
            self.assertEqualArray(exp, got)
        except AssertionError as e:
            rows = []
            for ra, rb in zip(exp, got):
                rows.append("--")
                rows.extend([str(ra.ravel()), str(rb.ravel())])
            raise AssertionError(
                "MISMATCH {}\n{}".format(inputs, "\n".join(rows))) from e

    def test_experimental_pad_positive(self):
        for verbose in [True, False]:
            with self.subTest(verbose=verbose):
                arr = numpy.arange(6) + 10
                paddings = numpy.array([1, 1]).reshape((-1, 2)) * 2
                self.fct_test(custom_pad, numpy.pad, arr,
                              paddings, verbose=verbose)

                arr = numpy.arange(6) + 10
                paddings = numpy.array([1, 1]).reshape((-1, 2))
                self.fct_test(custom_pad, numpy.pad, arr,
                              paddings, verbose=verbose)

                arr = numpy.arange(6).reshape((2, -1)) + 10
                paddings = numpy.array([1, 1, 1, 1]).reshape((-1, 2)) * 2
                self.fct_test(custom_pad, numpy.pad, arr,
                              paddings, verbose=verbose)

                arr = numpy.arange(6).reshape((2, -1)) + 10
                paddings = numpy.array([1, 1, 2, 2]).reshape((-1, 2))
                self.fct_test(custom_pad, numpy.pad, arr,
                              paddings, verbose=verbose)

                arr = numpy.arange(6).reshape((2, -1)) + 10
                paddings = numpy.array([1, 1, 1, 1]).reshape((-1, 2))
                self.fct_test(custom_pad, numpy.pad, arr,
                              paddings, verbose=verbose)

                arr = numpy.arange(6).reshape((1, 2, -1)) + 10
                paddings = numpy.array([1, 1, 1, 1, 1, 1]).reshape((-1, 2))
                self.fct_test(custom_pad, numpy.pad, arr,
                              paddings, verbose=verbose)

                arr = numpy.arange(6).reshape((1, 2, -1)) + 10
                paddings = numpy.array([1, 1, 1, 1, 1, 1]).reshape((-1, 2)) * 2
                self.fct_test(custom_pad, numpy.pad, arr,
                              paddings, verbose=verbose)

    def test_experimental_pad_552(self):
        arr = numpy.random.rand(2, 2, 2).astype(numpy.float32)
        paddings = numpy.array([1, 1, 1, 1, 1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

        arr = numpy.random.rand(5, 5, 2).astype(numpy.float32)
        paddings = numpy.array([1, 1, 1, 1, 1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

        arr = numpy.random.rand(2, 2, 2, 2).astype(numpy.float32)
        paddings = numpy.array([1, 1, 1, 1, 1, 1, 1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

        arr = numpy.random.rand(5, 5, 2).astype(numpy.float32)
        paddings = numpy.array([1, 1, 1, 1, 1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, self.ort_path_pad, arr, paddings)

    def test_experimental_pad_positive_ort(self):
        arr = (numpy.arange(6) + 10).astype(numpy.float32)
        paddings = numpy.array([1, 1]).reshape((-1, 2)) * 2
        self.fct_test(custom_pad, self.ort_path_pad, arr, paddings)

        arr = (numpy.arange(6) + 10).astype(numpy.float32)
        paddings = numpy.array([1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, self.ort_path_pad, arr, paddings)

        arr = (numpy.arange(6).reshape((2, -1)) + 10).astype(numpy.float32)
        paddings = numpy.array([1, 1, 1, 1]).reshape((-1, 2)) * 2
        self.fct_test(custom_pad, self.ort_path_pad, arr, paddings)

        arr = (numpy.arange(6).reshape((2, -1)) + 10).astype(numpy.float32)
        paddings = numpy.array([1, 1, 2, 2]).reshape((-1, 2))
        self.fct_test(custom_pad, self.ort_path_pad, arr, paddings)

        arr = (numpy.arange(6).reshape((2, -1)) + 10).astype(numpy.float32)
        paddings = numpy.array([1, 1, 1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, self.ort_path_pad, arr, paddings)

        arr = (numpy.arange(6).reshape((1, 2, -1)) + 10).astype(numpy.float32)
        paddings = numpy.array([1, 1, 1, 1, 1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, self.ort_path_pad, arr, paddings)

        arr = (numpy.arange(6).reshape((1, 2, -1)) + 10).astype(numpy.float32)
        paddings = numpy.array([1, 1, 1, 1, 1, 1]).reshape((-1, 2)) * 2
        self.fct_test(custom_pad, self.ort_path_pad, arr, paddings)

    def test_experimental_pad_negative(self):
        arr = numpy.arange(6) + 10
        paddings = numpy.array([1, -1]).reshape((-1, 2)) * 2
        self.assertRaise(lambda: custom_pad(
            arr, paddings), NotImplementedError)

    def test_experimental_einsum(self):
        eq = "bsnh,btnh->bnts"

        x = numpy.arange(8).reshape((1, 2, 2, 2))
        y = numpy.arange(8).reshape((1, 2, 2, 2)) + 100
        ein = numpy.einsum(eq, x, y)
        ein2 = custom_einsum(eq, x, y)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2)

        self.capture(lambda: custom_einsum(eq, x, y, verbose=True))

        x = numpy.random.rand(1, 8, 3, 5)
        y = numpy.random.rand(1, 8, 3, 5)
        bady1 = numpy.random.rand(2, 8, 3, 5)
        bady2 = numpy.random.rand(1, 8, 3, 6)
        ein = numpy.einsum(eq, x, y)
        self.assertRaise(lambda: custom_einsum(
            eq, x.astype(int), y), RuntimeError)
        self.assertRaise(lambda: custom_einsum(
            "bsnhj,btnh->bnts", x, y), ValueError)
        self.assertRaise(lambda: custom_einsum(
            "bsnh,btnhj->bnts", x, y), ValueError)
        self.assertRaise(lambda: custom_einsum(eq, x, bady1), ValueError)
        self.assertRaise(lambda: custom_einsum(eq, x, bady2), ValueError)
        self.assertRaise(lambda: custom_einsum(eq, bady1, x), ValueError)
        self.assertRaise(lambda: custom_einsum(eq, bady2, x), ValueError)
        self.assertRaise(
            lambda: custom_einsum(
                "bsnhv,btnhv->bnts", numpy.random.rand(1, 8, 3, 5, 2),
                numpy.random.rand(1, 8, 3, 5, 2)), NotImplementedError)
        ein2 = custom_einsum(eq, x, y)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2)

    def test_experimental_einsum_eq2(self):
        eq = "bshn,bthn->bnts"

        x = numpy.arange(8).reshape((1, 2, 2, 2))
        y = numpy.arange(8).reshape((1, 2, 2, 2)) + 100
        ein = numpy.einsum(eq, x, y)
        ein2 = custom_einsum(eq, x, y)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2)

        x = numpy.random.rand(1, 8, 3, 5)
        y = numpy.random.rand(1, 8, 3, 5)
        bady1 = numpy.random.rand(2, 8, 3, 5)
        bady2 = numpy.random.rand(1, 8, 3, 6)
        ein = numpy.einsum(eq, x, y)
        self.assertRaise(lambda: custom_einsum(
            eq, x.astype(int), y), RuntimeError)
        self.assertRaise(lambda: custom_einsum(
            "bsnhj,btnh->bnts", x, y), ValueError)
        self.assertRaise(lambda: custom_einsum(
            "bsnh,btnhj->bnts", x, y), ValueError)
        self.assertRaise(lambda: custom_einsum(eq, x, bady1), ValueError)
        self.assertRaise(lambda: custom_einsum(eq, x, bady2), ValueError)
        self.assertRaise(lambda: custom_einsum(eq, bady1, x), ValueError)
        self.assertRaise(lambda: custom_einsum(eq, bady2, x), ValueError)
        self.assertRaise(
            lambda: custom_einsum(
                "bsnhv,btnhv->bnts", numpy.random.rand(1, 8, 3, 5, 2),
                numpy.random.rand(1, 8, 3, 5, 2)), NotImplementedError)
        ein2 = custom_einsum(eq, x, y)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2)

    def is_ci_win(self):
        return is_travis_or_appveyor() == "appveyor"

    def test_experimental_einsum_c(self):
        eq = "bsnh,btnh->bnts"

        x = numpy.arange(8).reshape((1, 2, 2, 2)).astype(numpy.int64)
        y = (numpy.arange(8).reshape((1, 2, 2, 2)) + 100).astype(numpy.int64)
        ein = numpy.einsum(eq, x, y)
        ein2 = custom_einsum_int64(eq, x, y)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2)

        x = numpy.random.rand(1, 8, 3, 5)
        y = numpy.random.rand(1, 8, 3, 5)
        bady1 = numpy.random.rand(2, 8, 3, 5)
        bady2 = numpy.random.rand(1, 8, 3, 6)
        ein = numpy.einsum(eq, x, y)
        if not self.is_ci_win():
            # It crashes on appveyor.
            self.assertRaise(lambda: custom_einsum_double(
                "bsnhj,btnh->bnts", x, y), RuntimeError)
            self.assertRaise(lambda: custom_einsum_double(
                "bsnh,btnhj->bnts", x, y), RuntimeError)
            self.assertRaise(lambda: custom_einsum_double(
                eq, x, bady1), RuntimeError)
            self.assertRaise(lambda: custom_einsum_double(
                eq, x, bady2), RuntimeError)
            self.assertRaise(lambda: custom_einsum_double(
                eq, bady1, x), RuntimeError)
            self.assertRaise(lambda: custom_einsum_double(
                eq, bady2, x), RuntimeError)
            self.assertRaise(
                lambda: custom_einsum_double(
                    "bsnhv,btnhv->bnts", numpy.random.rand(1, 8, 3, 5, 2),
                    numpy.random.rand(1, 8, 3, 5, 2)), RuntimeError)
        ein2 = custom_einsum_double(eq, x, y)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2)

    def test_experimental_einsum_c_eq2(self):
        eq = "bshn,bthn->bnts"
        x = numpy.random.rand(1, 8, 3, 5)
        y = numpy.random.rand(1, 8, 3, 5)
        ein = numpy.einsum(eq, x, y)
        ein2 = custom_einsum_double(eq, x, y)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2)

    def test_experimental_einsum_c_eq2_optim(self):
        eq = "bsnh,btnh->bnts"
        x = numpy.random.rand(1, 8, 12, 64).astype(numpy.float64)
        y = numpy.random.rand(1, 8, 12, 64).astype(numpy.float64)
        ein = numpy.einsum(eq, x, y)
        ein2 = custom_einsum_float(eq, x, y)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2, decimal=5)

    def test_experimental_einsum_c_eq2_optim_th2(self):
        eq = "bsnh,btnh->bnts"

        x = numpy.arange(8).reshape((1, 2, 2, 2)).astype(numpy.int64)
        y = (numpy.arange(8).reshape((1, 2, 2, 2)) + 100).astype(numpy.int64)
        ein = numpy.einsum(eq, x, y)
        custom_einsum_int64(eq, x, y, nthread=1)
        ein2 = custom_einsum_int64(eq, x, y, nthread=2)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2)

        x = numpy.random.rand(1, 8, 12, 64).astype(numpy.float64)
        y = numpy.random.rand(1, 8, 12, 64).astype(numpy.float64)
        ein = numpy.einsum(eq, x, y)
        ein2 = custom_einsum_float(eq, x, y, 2)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2, decimal=5)

    def test_experimental_einsum_c_eq2_optim2(self):
        eq = "bshn,bthn->bnts"
        x = numpy.random.rand(1, 8, 12, 64).astype(numpy.float64)
        y = numpy.random.rand(1, 8, 12, 64).astype(numpy.float64)
        ein = numpy.einsum(eq, x, y)
        ein2 = custom_einsum_float(eq, x, y)
        self.assertEqual(ein.shape, ein2.shape)
        self.assertEqualArray(ein, ein2, decimal=5)

    def test_code_optimisation(self):
        res = code_optimisation()
        self.assertIn("=", res)

    def test_inplace_reduce_sum_rk_double(self):
        for i in [1, 2, 3, 4, 5, 6, 7, 11, 15, 23, 56, 99, 101, 128, 256]:
            with self.subTest(dim=i):
                mat = numpy.random.randn(i, i).astype(numpy.float64) * 100
                exp = mat.sum(axis=0).astype(numpy.float64)
                got = custom_reducesum_rk_double(mat)
                self.assertEqualArray(exp, got)

    def test_inplace_reduce_sum_rk_float(self):
        for i in [1, 2, 3, 4, 5, 6, 7, 11, 15, 23, 56, 99, 101, 128, 256]:
            with self.subTest(dim=i):
                mat = numpy.random.randn(i, i).astype(numpy.float32) * 100
                exp = mat.sum(axis=0).astype(numpy.float32)
                got = custom_reducesum_rk_float(mat, 1)
                self.assertEqualArray(exp, got)

    def test_inplace_reduce_sum_rk2(self):
        shape = (8, 4, 5)
        rnd = numpy.random.randn(*shape).astype(numpy.float64)
        mat = rnd.reshape((shape[0], -1))
        exp = mat.sum(axis=0).astype(numpy.float64)
        got = custom_reducesum_rk_double(mat)
        self.assertEqualArray(exp, got)

    def test_benchmark_cache(self):
        res = benchmark_cache(1000, False)
        self.assertGreater(res, 0)

    def test_benchmark_cache_tree(self):
        res = benchmark_cache_tree(1000)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2000)
        last = res[-1]
        self.assertEqual(last.trial, 1)
        self.assertEqual(last.row, 999)

        res = benchmark_cache_tree(12000)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 24000)


if __name__ == "__main__":
    TestExperimental().test_benchmark_cache_tree()
    unittest.main(verbosity=2)
