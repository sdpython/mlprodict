"""
@brief      test log(time=40s)
"""
import unittest
from numpy import array, float32
from onnx.helper import (
    make_model, make_node, set_model_props, make_graph,
    make_tensor_value_info)
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.onnx_backend import enumerate_onnx_tests
from mlprodict.onnxrt import OnnxInference


class TestOnnxBackEnd(ExtTestCase):

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
            print(len(missed), len(failed), len(mismatch))
            for t in failed:
                print("failed", str(t[0]).replace('\\\\', '\\'))
            for t in mismatch:
                print("mismatch", str(t[0]).replace('\\\\', '\\'))
            for t in missed:
                print("missed", str(t[0]).replace('\\\\', '\\'))

    def test_onnx_backend_test_to_python(self):
        name = 'test_abs'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_abs(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        if __name__ == '__main__':
            print(code[0])

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


if __name__ == "__main__":
    unittest.main()
