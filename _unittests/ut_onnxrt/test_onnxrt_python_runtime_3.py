"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node,
    make_graph, make_tensor_value_info, make_opsetid)
from onnx.checker import check_model
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict import get_ir_version, __max_supported_opset__ as TARGET_OPSET


class TestOnnxrtPythonRuntime3(ExtTestCase):

    def test_murmurhash3(self):
        for positive in [1, 0]:
            with self.subTest(positive=positive):
                X = make_tensor_value_info('X', TensorProto.STRING, [None])
                Y = make_tensor_value_info(
                    'Y',
                    TensorProto.UINT32 if positive == 1 else TensorProto.INT32,
                    [None])
                node = make_node('MurmurHash3', ['X'], ['Y'],
                                 domain="com.microsoft",
                                 positive=positive, seed=0)
                graph = make_graph([node], 'hash', [X], [Y])
                onnx_model = make_model(graph, opset_imports=[
                    make_opsetid('', TARGET_OPSET),
                    make_opsetid('com.microsoft', 1)])
                check_model(onnx_model)
            
                sess = OnnxInference(onnx_model, runtime="onnxruntime1")
                oinf = OnnxInference(onnx_model)

                # first try
                input_strings = ['a', 'aa', 'z0', 'o11',
                                 'd222', 'q4444', 't333', 'c5555',
                                 'z' * 100]
                as_bytes = [s.encode("utf-8") for s in input_strings]
                feeds = {'X': numpy.array(as_bytes)}
                expected = sess.run(feeds)
                got = oinf.run(feeds)
                
                self.assertEqual(expected['Y'].tolist()[:-1], got['Y'].tolist()[:-1])

                # second try
                input_strings = ['aa', 'a']
                as_bytes = [s.encode("utf-8") for s in input_strings]
                feeds = {'X': numpy.array(as_bytes)}
                expected = sess.run(feeds)
                got = oinf.run(feeds)
                
                self.assertEqual(expected['Y'].tolist()[1:], got['Y'].tolist()[1:])

    def test_murmurhash3_bug_ort(self):
        from onnxruntime import InferenceSession
        X = make_tensor_value_info('X', TensorProto.STRING, [None])
        Y = make_tensor_value_info('Y', TensorProto.UINT32, [None])
        node = make_node('MurmurHash3', ['X'], ['Y'],
                         domain="com.microsoft", positive=1, seed=0)
        graph = make_graph([node], 'hash', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[
            make_opsetid('', TARGET_OPSET),
            make_opsetid('com.microsoft', 1)])
        check_model(onnx_model)

        sess = InferenceSession(onnx_model.SerializeToString())
        x1 = numpy.array(['a', 'aa', 'z' * 100])
        x2 = numpy.array(['aa', 'a'])
        y1 = sess.run(None, {'X': x1})[0]
        y2 = sess.run(None, {'X': x2})[0]
        self.assertEqual(y1.tolist()[0], y2.tolist()[1])
        self.assertEqual(y1.tolist()[1], y2.tolist()[0])

        sess = InferenceSession(onnx_model.SerializeToString())
        x1 = numpy.array([b'a', b'aa', b'z' * 100])
        x2 = numpy.array([b'aa', b'a'])
        y1 = sess.run(None, {'X': x1})[0]
        y2 = sess.run(None, {'X': x2})[0]
        self.assertEqual(y1.tolist()[0], y2.tolist()[1])
        # fails
        # self.assertEqual(y1.tolist()[1], y2.tolist()[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
