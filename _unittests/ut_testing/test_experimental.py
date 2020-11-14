"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from onnx import helper, TensorProto
from onnxruntime import InferenceSession
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.experimental import custom_pad, custom_einsum
from mlprodict.tools import get_opset_number_from_onnx


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
        model = helper.make_model(graph, producer_name='model')
        op_set = model.opset_import[0]  # pylint: disable=E1101
        op_set.version = get_opset_number_from_onnx()
        sess = InferenceSession(model.SerializeToString())
        return numpy.squeeze(sess.run(['Y'], {'X': x, 'P': npads}))

    def fct_test(self, custom_fct, fct, *inputs):
        got = custom_fct(*inputs, debug=True)
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
        arr = numpy.arange(6) + 10
        paddings = numpy.array([1, 1]).reshape((-1, 2)) * 2
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

        arr = numpy.arange(6) + 10
        paddings = numpy.array([1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

        arr = numpy.arange(6).reshape((2, -1)) + 10
        paddings = numpy.array([1, 1, 1, 1]).reshape((-1, 2)) * 2
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

        arr = numpy.arange(6).reshape((2, -1)) + 10
        paddings = numpy.array([1, 1, 2, 2]).reshape((-1, 2))
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

        arr = numpy.arange(6).reshape((2, -1)) + 10
        paddings = numpy.array([1, 1, 1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

        arr = numpy.arange(6).reshape((1, 2, -1)) + 10
        paddings = numpy.array([1, 1, 1, 1, 1, 1]).reshape((-1, 2))
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

        arr = numpy.arange(6).reshape((1, 2, -1)) + 10
        paddings = numpy.array([1, 1, 1, 1, 1, 1]).reshape((-1, 2)) * 2
        self.fct_test(custom_pad, numpy.pad, arr, paddings)

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


if __name__ == "__main__":
    TestExperimental().test_experimental_einsum()
    unittest.main()
