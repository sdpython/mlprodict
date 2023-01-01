# pylint: disable=R1716
"""
@brief      test log(time=5s)
"""
import unittest
import numpy
from onnx import TensorProto, checker
from onnx.checker import check_model
from onnx.helper import (  # pylint: disable=W0611
    make_function, make_graph, make_model, make_node,
    make_opsetid, make_sequence_type_proto, make_tensor,
    make_tensor_sequence_value_info, make_tensor_value_info,
    make_value_info)
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.experimental.op_im2col import im2col
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.ops_onnx.op_conv import Conv


class TestReferenceEvaluator(ExtTestCase):

    def test_conv(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [
                                   None, None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [
                                   None, None, None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [
                                   None, None, None, None])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3])
        node = make_node(
            "Conv", ["X", "W", "B"], ["Y"], pads=[1, 1, 1, 1],
            dilations=[1, 1], strides=[2, 2], kernel_shape=[3, 3])
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        sess1 = ReferenceEvaluator(onnx_model, new_ops=[Conv])
        sess2 = ReferenceEvaluator(onnx_model)

        sH, sW = 5, 6
        for i in range(sH):
            for j in range(sW):
                X = numpy.zeros((1, 1, sH, sW), dtype=numpy.float32)
                X[0, 0, i, j] = 1.0
                W = numpy.zeros((1, 1, 3, 3), dtype=numpy.float32)
                W[0, 0, :, :] = numpy.minimum(
                    2 ** numpy.arange(9).reshape((3, -1)), 256)

                B = numpy.array([[[[0]]]], dtype=numpy.float32)
                expected = sess1.run(None, {"X": X, "W": W, "B": B})[0]
                got = sess2.run(None, {"X": X, "W": W, "B": B})[0]
                self.assertEqualArray(expected, got)
        self.assertEqual(len(sess1.rt_nodes_[0]._cache), 1)

    def test_conv_none(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [
                                   None, None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [
                                   None, None, None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [
                                   None, None, None, None])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [
                                   None, None, None, None])
        node = make_node(
            "Conv", ["X", "W", "B"], ["Y"], pads=[1, 1, 1, 1],
            dilations=[1, 1], strides=[2, 2])
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        sess1 = ReferenceEvaluator(onnx_model, new_ops=[Conv])
        sess2 = ReferenceEvaluator(onnx_model)

        sH, sW = 5, 6
        for i in range(sH):
            for j in range(sW):
                X = numpy.zeros((1, 1, sH, sW), dtype=numpy.float32)
                X[0, 0, i, j] = 1.0
                W = numpy.zeros((1, 1, 3, 3), dtype=numpy.float32)
                W[0, 0, :, :] = numpy.minimum(
                    2 ** numpy.arange(9).reshape((3, -1)), 256)

                B = numpy.array([[[[0]]]], dtype=numpy.float32)
                expected = sess1.run(None, {"X": X, "W": W, "B": B})[0]
                got = sess2.run(None, {"X": X, "W": W, "B": B})[0]
                self.assertEqualArray(expected, got)
        self.assertEqual(len(sess1.rt_nodes_[0]._cache), 1)

    def test_conv_im2col_group4(self):
        # model 1
        X = make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 6, 6])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [4, 1, 3, 3])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [4])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 6, 6])

        node = make_node(
            "Conv",
            ["X", "W", "B"],
            ["Y"],
            group=4,
            dilations=[1, 1],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        feeds = {
            "X": numpy.arange(2 * 4 * 6 * 6).reshape((2, 4, 6, 6)).astype(numpy.float32),
            "W": numpy.array([[[[-0.026239916682243347,
                                0.07565222680568695,
                                -0.03209298849105835,
                                 ],
                                [
                                -0.08708783239126205,
                                0.0961190015077591,
                                0.13418219983577728,
                                ],
                                [
                                0.1598859578371048,
                                0.03840477764606476,
                                -0.13170936703681946,
                                ],
                                ]
                               ],
                              [
                [
                    [
                        -0.0689004510641098,
                        0.1408083587884903,
                        -0.03717087209224701,
                    ],
                    [
                        0.030967697501182556,
                        0.0263785719871521,
                        -0.0899493545293808,
                    ],
                    [
                        0.07828782498836517,
                        -0.06266771256923676,
                        0.10750330984592438,
                    ],
                ]
            ],
                [
                [
                    [
                        0.020227551460266113,
                        -0.04353883117437363,
                        -0.10938453674316406,
                    ],
                    [
                        -0.14101561903953552,
                        -0.03393106162548065,
                        0.12139306962490082,
                    ],
                    [
                        0.02838282287120819,
                        0.13864465057849884,
                        -0.06065710633993149,
                    ],
                ]
            ],
                [
                [
                    [
                        -0.06511610746383667,
                        -0.05987360328435898,
                        -0.008047685027122498,
                    ],
                    [
                        0.07340313494205475,
                        0.0326494425535202,
                        0.012516498565673828,
                    ],
                    [
                        0.13260947167873383,
                        -0.022225692868232727,
                        -0.11167611926794052,
                    ],
                ]
            ],
            ],
                dtype=numpy.float32,
            ),
            "B": numpy.array([-0.1457933485507965, -0.07481209933757782,
                              -0.05890338122844696, -0.11964251846075058],
                             dtype=numpy.float32),
        }
        feeds["B"][:] = 0

        # model 2
        X = feeds["X"]
        W = feeds["W"]
        B = feeds["B"]
        Y = numpy.empty((2, 4, 6, 6), dtype=X.dtype)
        for b in range(X.shape[0]):
            for g in range(4):
                x = X[b: b + 1, g: g + 1]
                w = W[g]
                c2 = im2col(x, (3, 3), [1, 1], [1, 1, 1, 1], [1, 1])
                mul = numpy.matmul(c2, w.flatten())
                mul = mul + B[g]
                Y[b, g, :, :] = mul

        ref1 = ReferenceEvaluator(onnx_model, new_ops=[Conv])
        got1 = ref1.run(None, feeds)

        self.assertEqualArray(Y, got1[0], atol=1e-5)

    def test_conv_strides(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 6, 6])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [2, 3, 3, 3])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [2])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [
                                   None, None, None, None])

        node = make_node(
            "Conv",
            ["X", "W", "B"],
            ["Y"],
            group=1,
            dilations=[1, 1],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        )
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        feeds = {
            "X": numpy.arange(1 * 3 * 6 * 6).reshape((1, 3, 6, 6)).astype(numpy.float32) + 1,
            "W": numpy.zeros((2, 3, 3, 3), dtype=numpy.float32),
            "B": numpy.zeros((2,), dtype=numpy.float32),
        }
        feeds["W"][0, 0, 0, 1] = 1

        ref1 = ReferenceEvaluator(onnx_model, new_ops=[Conv])
        got1 = ref1.run(None, feeds)
        expected = numpy.array(
            [
                [
                    [[0.0, 0.0, 0.0], [7.0, 9.0, 11.0], [19.0, 21.0, 23.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ],
            dtype=numpy.float32,
        )

        self.assertEqualArray(expected, got1[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
