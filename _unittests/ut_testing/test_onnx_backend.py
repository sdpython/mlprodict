"""
@brief      test log(time=10s)
"""
import os
import unittest
from numpy import array, float32, int64
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_graph,
    make_tensor_value_info, __file__ as onnx_file)
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.onnx_backend import enumerate_onnx_tests
from mlprodict.onnxrt import OnnxInference


class TestOnnxBackEnd(ExtTestCase):

    def test_onnx_backend_test_to_python(self):
        name = 'test_abs'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_abs(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])

    @staticmethod
    def load_fct(obj, runtime='python'):
        return OnnxInference(obj, runtime)

    @staticmethod
    def run_fct(obj, *inputs):
        names = obj.input_names
        if len(names) < len(inputs):
            raise AssertionError(
                "Got %d inputs but expecting %d." % (
                    len(inputs), len(names)))
        feeds = {names[i]: inputs[i] for i in range(len(inputs))}
        got = obj.run(feeds)

        names = obj.output_names
        return [got[n] for n in names]

    def test_enumerate_onnx_tests_run_one(self):
        done = 0
        for te in enumerate_onnx_tests('node', lambda folder: folder == 'test_abs'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_tests_run(self):

        self.assertRaise(lambda: list(
            enumerate_onnx_tests('NNN')), FileNotFoundError)
        missed = []
        failed = []
        mismatch = []
        for te in enumerate_onnx_tests('node'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            try:
                te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            except NotImplementedError as e:
                missed.append((te, e))
                continue
            except (IndexError, RuntimeError, TypeError, ValueError,
                    AttributeError) as e:
                failed.append((te, e))
                continue
            except AssertionError as e:
                mismatch.append((te, e))
                continue

        if __name__ == '__main__':
            path = os.path.dirname(onnx_file)
            print(len(missed), len(failed), len(mismatch))
            for t in failed:
                print("failed",
                      str(t[0]).replace('\\\\', '\\').replace(
                        path, 'onnx').replace("\\", "/"))
            for t in mismatch:
                print("mismatch",
                      str(t[0]).replace('\\\\', '\\').replace(
                        path, 'onnx').replace("\\", "/"))
            for t in missed:
                print("missed",
                      str(t[0]).replace('\\\\', '\\').replace(
                        path, 'onnx').replace("\\", "/"))

    def test_abs(self):

        def create_model():
            '''
            Converted ``test_abs``.

            * producer: backend-test
            * version: 0
            * description:
            '''

            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 9}

            value = make_tensor_value_info('x', 1, [3, 4, 5])
            inputs.append(value)

            value = make_tensor_value_info('y', 1, [3, 4, 5])
            outputs.append(value)

            node = make_node(
                'Abs',
                ['x'],
                ['y'],
                domain='')
            nodes.append(node)

            graph = make_graph(nodes, 'test_abs', inputs,
                               outputs, initializers)

            onnx_model = make_model(graph)
            onnx_model.ir_version = 3
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})

            del onnx_model.opset_import[:]  # pylint: disable=E1101
            for dom, value in opsets.items():
                op_set = onnx_model.opset_import.add()
                op_set.domain = dom
                op_set.version = value

            return onnx_model

        onnx_model = create_model()

        oinf = OnnxInference(onnx_model)
        xs = [
            array([[[1.7640524, 0.4001572, 0.978738, 2.2408931,
                     1.867558],
                    [-0.9772779, 0.95008844, -0.1513572, -0.10321885,
                     0.41059852],
                    [0.14404356, 1.4542735, 0.7610377, 0.12167501,
                     0.44386324],
                    [0.33367434, 1.4940791, -0.20515826, 0.3130677,
                     -0.85409576]],

                   [[-2.5529897, 0.6536186, 0.8644362, -0.742165,
                     2.2697546],
                    [-1.4543657, 0.04575852, -0.18718386, 1.5327792,
                     1.4693588],
                    [0.15494743, 0.37816253, -0.88778573, -1.9807965,
                     -0.34791216],
                    [0.15634897, 1.2302907, 1.2023798, -0.3873268,
                     -0.30230275]],

                   [[-1.048553, -1.420018, -1.7062702, 1.9507754,
                     -0.5096522],
                    [-0.4380743, -1.2527953, 0.7774904, -1.6138978,
                     -0.21274029],
                    [-0.89546657, 0.3869025, -0.51080513, -1.1806322,
                     -0.02818223],
                    [0.42833188, 0.06651722, 0.3024719, -0.6343221,
                     -0.36274117]]], dtype=float32),
        ]
        ys = [
            array([[[1.7640524, 0.4001572, 0.978738, 2.2408931, 1.867558],
                    [0.9772779, 0.95008844, 0.1513572, 0.10321885, 0.41059852],
                    [0.14404356, 1.4542735, 0.7610377, 0.12167501, 0.44386324],
                    [0.33367434, 1.4940791, 0.20515826, 0.3130677, 0.85409576]],

                   [[2.5529897, 0.6536186, 0.8644362, 0.742165, 2.2697546],
                    [1.4543657, 0.04575852, 0.18718386, 1.5327792, 1.4693588],
                    [0.15494743, 0.37816253, 0.88778573, 1.9807965, 0.34791216],
                    [0.15634897, 1.2302907, 1.2023798, 0.3873268, 0.30230275]],

                   [[1.048553, 1.420018, 1.7062702, 1.9507754, 0.5096522],
                    [0.4380743, 1.2527953, 0.7774904, 1.6138978, 0.21274029],
                    [0.89546657, 0.3869025, 0.51080513, 1.1806322, 0.02818223],
                    [0.42833188, 0.06651722, 0.3024719, 0.6343221, 0.36274117]]],
                  dtype=float32),
        ]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_onnx_backend_test_to_python_argmax(self):
        name = 'test_argmax_negative_axis_keepdims_example'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_argmax_negative_axis_keepdims_example(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #     print(code[0])

    def test_argmax_negative_axis_keepdims_example(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 11}
            target_opset = 11  # subgraphs

            value = make_tensor_value_info('data', 1, [2, 2])
            inputs.append(value)

            value = make_tensor_value_info('result', 7, [2, 1])
            outputs.append(value)

            node = make_node('ArgMax', ['data'], ['result'],
                             axis=-1, keepdims=1, domain='')
            nodes.append(node)

            graph = make_graph(
                nodes, 'test_argmax_negative_axis_keepdims_example',
                inputs, outputs, initializers)

            onnx_model = make_model(graph)
            onnx_model.ir_version = 6
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})

            del onnx_model.opset_import[:]  # pylint: disable=E1101
            for dom, value in opsets.items():
                op_set = onnx_model.opset_import.add()
                op_set.domain = dom
                op_set.version = value

            return onnx_model

        onnx_model = create_model()

        oinf = OnnxInference(onnx_model)
        xs = [array([[2., 1.], [3., 10.]], dtype=float32)]
        ys = [array([[0], [1]], dtype=int64)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_onnx_backend_test_cast_FLOAT_to_STRING(self):
        name = 'test_cast_FLOAT_to_STRING'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_cast_FLOAT_to_STRING(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #      print(code[0])

    def test_cast_FLOAT_to_STRING(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 10}
            target_opset = 10  # subgraphs

            inputs.append(make_tensor_value_info('input', 1, [3, 4]))
            outputs.append(make_tensor_value_info('output', 8, [3, 4]))
            nodes.append(make_node('Cast', ['input'], ['output'],
                                   to=TensorProto.STRING, domain=''))
            graph = make_graph(nodes, 'test_cast_FLOAT_to_STRING',
                               inputs, outputs, initializers)

            onnx_model = make_model(graph)
            onnx_model.ir_version = 4
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})

            del onnx_model.opset_import[:]  # pylint: disable=E1101
            for dom, value in opsets.items():
                op_set = onnx_model.opset_import.add()
                op_set.domain = dom
                op_set.version = value

            return onnx_model

        onnx_model = create_model()

        oinf = OnnxInference(onnx_model)
        xs = [array([[0.9767611, 0.6048455, 0.7392636, 0.039187793],
                     [0.28280696, 0.12019656, 0.2961402, 0.11872772],
                     [0.31798318, 0.41426298, 0.064147495, 0.6924721]],
                    dtype=float32)]
        ys = [array([['0.9767611', '0.6048455', '0.7392636', '0.039187793'],
                     ['0.28280696', '0.12019656', '0.2961402', '0.11872772'],
                     ['0.31798318', '0.41426298', '0.064147495', '0.6924721']],
                    dtype=object)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqual(y.tolist(), gy.tolist())


if __name__ == "__main__":
    # TestOnnxBackEnd().test_cast_FLOAT_to_STRING()
    unittest.main()
