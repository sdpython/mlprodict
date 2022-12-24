"""
@brief      test log(time=10s)
"""
import os
import unittest
import numpy
from numpy import array, float32, int64, int8, int32, uint8
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator
from onnx.helper import (
    make_model, make_node, set_model_props, make_graph,
    make_tensor_value_info, make_opsetid, make_tensor,
    __file__ as onnx_file)
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.testing.onnx_backend import (
    enumerate_onnx_tests, assert_almost_equal_string)
from mlprodict.onnxrt import OnnxInference


class Evaluator(ReferenceEvaluator):
    def run(self, feeds):
        res = ReferenceEvaluator.run(self, None, feeds)
        return dict(zip(self.output_names, res))


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
        if runtime == 'python':
            try:
                return OnnxInference(obj, runtime)
            except Exception as e:
                raise AssertionError(f"Unable to load model {obj}.") from e
        if runtime == "onnx":
            verbose = 0
            try:
                return Evaluator(obj, verbose=verbose)
            except Exception as e:
                raise AssertionError(f"Unable to load model {obj}.") from e
        raise NotImplementedError(f"Unknown runtime={runtime!r}.")

    @staticmethod
    def run_fct(obj, *inputs):
        names = obj.input_names
        if len(names) < len(inputs):
            raise AssertionError(
                f"Got {len(inputs)} inputs but expecting {len(names)}.")
        feeds = {names[i]: inputs[i].copy()
                 for i in range(len(inputs))}
        got = obj.run(feeds)

        names = obj.output_names
        return [got[n] for n in names]

    def test_enumerate_onnx_tests_run_one(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node',
                lambda folder: folder == 'test_bitwise_not_3d'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(lambda *args: TestOnnxBackEnd.load_fct(*args, runtime='onnx'),
                   TestOnnxBackEnd.run_fct)
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
        self.assertIn(
            'def test_argmax_negative_axis_keepdims_example(self):', code[0])
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
        #       print(code[0])

    @ignore_warnings(DeprecationWarning)
    def test_cast_FLOAT_to_STRING(self):
        from numpy import object as dtype_object

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 10}

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
        xs = [array([[0.9767611, 0.6048455, 0.7392636, 0.03918779],
                     [0.28280696, 0.12019656, 0.2961402, 0.11872772],
                     [0.31798318, 0.41426298, 0.06414749, 0.6924721]],
                    dtype=float32)]
        ys = [array([['0.9767611', '0.6048455', '0.7392636', '0.039187793'],
                     ['0.28280696', '0.12019656', '0.2961402', '0.11872772'],
                     ['0.31798318', '0.41426298', '0.064147495', '0.6924721']],
                    dtype=object)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            if y.dtype == dtype_object:
                assert_almost_equal_string(y, gy)
            else:
                raise AssertionError("dtype is wrong.")

    def test_onnx_backend_test_logsoftmax_axis_0(self):
        name = 'test_logsoftmax_axis_0'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_logsoftmax_axis_0(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #    print(code[0])

    def test_logsoftmax_axis_0(self):

        def create_model():

            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 13}

            inputs.append(make_tensor_value_info('x', 1, [3, 4, 5]))
            outputs.append(make_tensor_value_info('y', 1, [3, 4, 5]))
            nodes.append(make_node('LogSoftmax', [
                         'x'], ['y'], axis=0, domain=''))
            graph = make_graph(nodes, 'test_logsoftmax_axis_0',
                               inputs, outputs, initializers)

            onnx_model = make_model(graph)
            onnx_model.ir_version = 7
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
        ys = [
            array([[[-1.3056276, -1.6216207, -1.3767376, -0.6788401, -1.0124384],
                    [-1.161458, -1.0146257, -1.362729, -2.272813, -1.5482603],
                    [-1.4185143, -0.52166486, -1.0694411, -2.3322854, -0.94328284],
                    [-1.077317, -0.69715375, -1.5713093, -1.2400951, -0.7828569]],

                   [[-0.5166902, -1.3681593, -1.4910393, -2.1775682, -0.6102418],
                    [-0.68437016, -1.9189556, -1.3269023, -0.8432526, -0.48950005],
                    [-1.4076104, -1.5977758, -0.9426931, -0.47316402, -1.0392339],
                    [-1.2546424, -0.9609422, -0.57408774, -1.1658361, -1.3346498]],

                   [[-2.021127, -0.6017599, -0.6492053, -0.96895784, -2.3703442],
                    [-1.7006615, -0.7119188, -0.73659575, -0.762134, -1.7461185],
                    [-0.66709125, -1.5890357, -1.3196738, -1.2733283, -1.3589638],
                    [-0.98265946, -2.1247156, -1.4739957, -0.91884077, -1.2742114]]],
                  dtype=float32),
        ]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy, decimal=6)

    def test_onnx_backend_test_averagepool_2d_ceil(self):
        name = 'test_averagepool_2d_ceil'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_averagepool_2d_ceil(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #     print(code[0])

    def test_averagepool_2d_ceil(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 10}
            inputs.append(make_tensor_value_info('x', 1, [1, 1, 4, 4]))
            outputs.append(make_tensor_value_info('y', 1, [1, 1, 2, 2]))

            node = make_node(
                'AveragePool', ['x'], ['y'],
                ceil_mode=1, kernel_shape=[3, 3], strides=[2, 2], domain='')
            nodes.append(node)

            graph = make_graph(nodes, 'test_averagepool_2d_ceil',
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
        xs = [array([[[[1., 2., 3., 4.],
                       [5., 6., 7., 8.],
                       [9., 10., 11., 12.],
                       [13., 14., 15., 16.]]]], dtype=float32)]
        ys = [array([[[[6., 7.5],
                       [12., 13.5]]]], dtype=float32)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_onnx_backend_test_batchnorm_epsilon_training_mode(self):
        name = 'test_batchnorm_epsilon_training_mode'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn(
            'def test_batchnorm_epsilon_training_mode(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #     print(code[0])

    def test_batchnorm_epsilon_training_mode(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 14}

            inputs.append(make_tensor_value_info('x', 1, [2, 3, 4, 5]))
            inputs.append(make_tensor_value_info('s', 1, [3]))
            inputs.append(make_tensor_value_info('bias', 1, [3]))
            inputs.append(make_tensor_value_info('mean', 1, [3]))
            inputs.append(make_tensor_value_info('var', 1, [3]))
            outputs.append(make_tensor_value_info('y', 1, [2, 3, 4, 5]))
            outputs.append(make_tensor_value_info('output_mean', 1, [3]))
            outputs.append(make_tensor_value_info('output_var', 1, [3]))

            node = make_node(
                'BatchNormalization',
                ['x', 's', 'bias', 'mean', 'var'],
                ['y', 'output_mean', 'output_var'],
                epsilon=0.009999999776482582, training_mode=1, domain='')
            nodes.append(node)

            graph = make_graph(
                nodes, 'test_batchnorm_epsilon_training_mode', inputs, outputs, initializers)

            onnx_model = make_model(graph)
            onnx_model.ir_version = 7
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
            array([[[[0.40746182, 1.3439544, -0.818221, 0.08270994, -1.2910584],
                     [-0.6611042, -1.180191, 0.19764264, 0.4139, 1.197322],
                     [1.8833538, 0.7142238, 2.2843335, 1.5641025, 0.6111037],
                     [-0.8773633, -1.6210876, -0.581673, -0.5378339, -1.5560237]],

                    [[-0.05446484, -1.8112788, -0.6311752, -0.9281592, 1.490722],
                     [0.19549933, -0.47160435, 1.8123547, -2.2941375, 0.65120935],
                     [-1.1304965, -0.7773467, 1.1159384, 1.339453, -1.7674336],
                     [0.42441246, 1.0893091, -0.38418567, 0.6322014, -0.5496559]],

                    [[0.52112573, 0.10834955, 0.26166847, -0.91475534, 0.8582378],
                     [0.09433429, -1.4859039, -1.9005842, -1.1375792, -1.7620388],
                     [-0.2886232, 1.0479822, 0.24995755, 0.04690446, -1.032243],
                     [0.4031857, -0.68405926, 1.2623222, -2.0055566, -0.3320304]]],

                   [[[-0.2961004, -2.2183607, -0.18350288, 0.39230806, 0.2416348],
                     [0.10393591, -0.8295712, 0.49275938, 0.09011279, -0.99756753],
                     [-0.8000382, 0.20707558, 0.523463, -0.6993948, 0.9137058],
                     [-0.6727848, 0.1333245, 0.426896, -0.01284939, -0.3522483]],

                    [[0.8194666, 0.52198774, 1.1972599, -0.38248622, 0.6916619],
                     [0.35388502, 1.0475854, -0.42389622, -3.5147681, -1.3431567],
                     [1.4255061, 0.22858201, -0.25766376, 0.05037072, -1.3802109],
                     [-0.26167208, -0.17937969, -0.6927706, 1.1378269, -0.16915725]],

                    [[-0.7639137, -0.4980731, -0.3628911, 0.2639603, -0.6296419],
                     [-0.47225842, -1.5133611, 1.1076247, 0.17623875, -0.9403535],
                     [0.92959434, -1.0627949, -0.88640624, 1.9213469, -0.4597805],
                     [-1.0890344, 0.98411727, -1.1592063, -0.4365371, 1.0092446]]]], dtype=float32),
            array([0.7133896, -0.72805774, 0.83951646], dtype=float32),
            array([1.239021, -1.7848039, -0.79618585], dtype=float32),
            array([-1.4005413, -0.18435058, -1.3911932], dtype=float32),
            array([0.0446123, 0.79979587, 0.07695644], dtype=float32),
        ]
        ys = [
            array([[[[1.578124, 2.2737765, 0.6676531, 1.3368894, 0.31641638],
                     [0.78436375, 0.3987717, 1.4222646, 1.5829065, 2.164854],
                     [2.6744573, 1.8059952, 2.972316, 2.4373088, 1.7293949],
                     [0.62372047, 0.07126164, 0.8433674, 0.8759323, 0.11959279]],

                    [[-1.8009548, -0.66743493, -1.4288535, -1.2372355, -2.7979298],
                     [-1.962235, -1.5318108, -3.0054517, -0.3558879, -2.2562652],
                     [-1.1066847, -1.3345418, -2.5561147, -2.7003293, -0.69572437],
                     [-2.109933, -2.538933, -1.5882145, -2.244001, -1.4814509]],

                    [[-0.10020548, -0.46598074, -0.33011955, -1.3725895, 0.19852114],
                     [-0.47840014, -1.878704, -2.2461667, -1.5700414, -2.1233969],
                     [-0.81775206, 0.36666036, -0.340497, -0.5204294, -1.4766994],
                     [-0.2047162, -1.1681616, 0.5565944, -2.3391862, -0.85621667]]],

                   [[[1.0554986, -0.37240934, 1.1391392, 1.5668674, 1.4549432],
                     [1.3526566, 0.6592218, 1.6414853, 1.3423884, 0.5344295],
                     [0.68115973, 1.4292716, 1.6642929, 0.7559204, 1.9541761],
                     [0.7756871, 1.3744873, 1.5925603, 1.2659053, 1.0137904]],

                    [[-2.364827, -2.1728897, -2.6085844, -1.589311, -2.2823658],
                     [-2.0644276, -2.5120122, -1.5625927, 0.43167925, -0.96947354],
                     [-2.7558517, -1.9835804, -1.6698481, -1.8685961, -0.9455657],
                     [-1.6672618, -1.720358, -1.3891113, -2.5702374, -1.7269537]],

                    [[-1.2389234, -1.0033529, -0.8835634, -0.32808864, -1.1199405],
                     [-0.9804776, -1.9030348, 0.41951156, -0.4058218, -1.395273],
                     [0.26175272, -1.5037725, -1.3474684, 1.1405791, -0.9694205],
                     [-1.5270243, 0.31006742, -1.589206, -0.9488237, 0.33233356]]]], dtype=float32),
            array([-1.2653913, -0.17386518, -1.2785023], dtype=float32),
            array([0.1313822, 0.84614456, 0.15801588], dtype=float32),
        ]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy, atol=1e-6)

    def test_onnx_backend_test_clip_default_int8_inbounds(self):
        name = 'test_clip_default_int8_inbounds'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_clip_default_int8_inbounds(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #     print(code[0])

    def test_clip_default_int8_inbounds(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 12}
            inputs.append(make_tensor_value_info('x', 3, [3]))
            outputs.append(make_tensor_value_info('y', 3, [3]))
            nodes.append(make_node('Clip', ['x'], ['y'], domain=''))
            graph = make_graph(nodes, 'test_clip_default_int8_inbounds',
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
        xs = [array([-1, 0, 1], dtype=int8)]
        ys = [array([-1, 0, 1], dtype=int8)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_onnx_backend_test_einsum_inner_prod(self):
        name = 'test_einsum_inner_prod'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_einsum_inner_prod(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #    print(code[0])

    def test_einsum_inner_prod(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 12}
            inputs.append(make_tensor_value_info('x', 11, [5]))
            inputs.append(make_tensor_value_info('y', 11, [5]))
            outputs.append(make_tensor_value_info('z', 11, None))
            node = make_node('Einsum', ['x', 'y'], ['z'], equation=b'i,i',
                             domain='')
            nodes.append(node)
            graph = make_graph(nodes, 'test_einsum_inner_prod',
                               inputs, outputs, initializers)

            onnx_model = make_model(graph)
            onnx_model.ir_version = 7
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
        xs = [array([1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799]),
              array([-0.97727788, 0.95008842, -0.15135721, -0.10321885, 0.4105985])]
        ys = [array(-0.95640957)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_onnx_backend_test_identity_opt(self):
        name = 'test_identity_opt'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_identity_opt(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #     print(code[0])

    def test_identity_opt(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 16}

            inputs.append(make_tensor_value_info('opt_in', 0, None))
            outputs.append(make_tensor_value_info('opt_out', 0, None))
            node = make_node('Identity', ['opt_in'], ['opt_out'], domain='')
            nodes.append(node)
            graph = make_graph(nodes, 'test_identity_opt',
                               inputs, outputs, initializers)

            onnx_model = make_model(graph)
            onnx_model.ir_version = 8
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
        xs = []
        ys = []
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_onnx_backend_test_identity_sequence(self):
        name = 'test_identity_sequence'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn('def test_identity_sequence(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #    print(code[0])

    def test_identity_sequence(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 16}
            inputs.append(make_tensor_value_info('x', 0, None))
            outputs.append(make_tensor_value_info('y', 0, None))
            nodes.append(make_node('Identity', ['x'], ['y'], domain=''))
            opset_imports = [make_opsetid(domain, version)
                             for domain, version in opsets.items()]
            graph = make_graph(nodes, 'test_identity_sequence',
                               inputs, outputs, initializers)
            onnx_model = make_model(graph, opset_imports=opset_imports)
            onnx_model.ir_version = 8
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})

            return onnx_model

        onnx_model = create_model()

        oinf = OnnxInference(onnx_model)
        xs = [[array([[[[1., 2.], [3., 4.]]]], dtype=float32),
               array([[[[2., 3.], [1., 5.]]]], dtype=float32)]]
        ys = [[array([[[[1., 2.], [3., 4.]]]], dtype=float32),
               array([[[[2., 3.], [1., 5.]]]], dtype=float32)]]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_onnx_backend_test_gather_elements_negative_indices(self):
        name = 'test_gather_elements_negative_indices'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn(
            'def test_gather_elements_negative_indices(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #   print(code[0])

    def test_gather_elements_negative_indices(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 11}
            inputs.append(make_tensor_value_info('data', 1, [3, 3]))
            inputs.append(make_tensor_value_info('indices', 7, [2, 3]))
            outputs.append(make_tensor_value_info('y', 1, [2, 3]))
            node = make_node(
                'GatherElements', ['data', 'indices'], ['y'], axis=0, domain='')
            nodes.append(node)
            opset_imports = [make_opsetid(domain, version)
                             for domain, version in opsets.items()]
            graph = make_graph(
                nodes, 'test_gather_elements_negative_indices', inputs, outputs, initializers)
            onnx_model = make_model(graph, opset_imports=opset_imports)
            onnx_model.ir_version = 6
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})
            return onnx_model

        onnx_model = create_model()
        oinf = OnnxInference(onnx_model)
        xs = [array([[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]], dtype=float32),
              array([[-1, -2, 0],
                     [-2, 0, 0]], dtype=int64)]
        ys = [array([[7., 5., 3.],
                     [4., 2., 3.]], dtype=float32)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_onnx_backend_test_constantofshape_int_shape_zero(self):
        name = 'test_constantofshape_int_shape_zero'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn(
            'def test_constantofshape_int_shape_zero(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #    print(code[0])

    def test_constantofshape_int_shape_zero(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 12}
            inputs.append(make_tensor_value_info('x', 7, [1]))
            outputs.append(make_tensor_value_info('y', 6, [None]))
            node = make_node(
                'ConstantOfShape', ['x'], ['y'],
                value=make_tensor("value", TensorProto.INT32,
                                  dims=[1], vals=[0]),
                domain='')
            nodes.append(node)
            opset_imports = [make_opsetid(domain, version)
                             for domain, version in opsets.items()]
            graph = make_graph(
                nodes, 'test_constantofshape_int_shape_zero',
                inputs, outputs, initializers)

            onnx_model = make_model(graph, opset_imports=opset_imports)
            onnx_model.ir_version = 6
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})
            return onnx_model

        onnx_model = create_model()
        oinf = OnnxInference(onnx_model)
        xs = [array([0], dtype=int64)]
        ys = [array([], dtype=int32)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_onnx_backend_test_reduce_sum_default_axes_keepdims_example(self):
        name = 'test_reduce_sum_default_axes_keepdims_example'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn(
            'def test_reduce_sum_default_axes_keepdims_example(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #    print(code[0])

    def test_reduce_sum_default_axes_keepdims_example(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []

            opsets = {'': 13}
            inputs.append(make_tensor_value_info('data', 1, [3, 2, 2]))
            inputs.append(make_tensor_value_info('axes', 7, [None]))
            outputs.append(make_tensor_value_info('reduced', 1, [1, 1, 1]))
            node = make_node('ReduceSum', ['data', 'axes'], ['reduced'],
                             keepdims=1, domain='')
            nodes.append(node)
            opset_imports = [make_opsetid(domain, version)
                             for domain, version in opsets.items()]
            graph = make_graph(
                nodes, 'test_reduce_sum_default_axes_keepdims_example', inputs, outputs, initializers)
            onnx_model = make_model(graph, opset_imports=opset_imports)
            onnx_model.ir_version = 7
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})
            return onnx_model

        onnx_model = create_model()
        oinf = OnnxInference(onnx_model)
        xs = [array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
                     [[9., 10.], [11., 12.]]], dtype=float32),
              array([], dtype=int64)]
        ys = [array([[[78.]]], dtype=float32)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_enumerate_onnx_tests_test_clip_default_inbounds(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_clip_default_inbounds'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_onnx_backend_test_bernoulli(self):
        name = 'test_bernoulli'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn(
            'def test_bernoulli(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #    print(code[0])

    def test_bernoulli(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []
            functions = []

            opsets = {'': 15}
            inputs.append(make_tensor_value_info('x', 11, [10]))
            outputs.append(make_tensor_value_info('y', 11, [10]))
            node = make_node('Bernoulli', ['x'], ['y'], domain='')
            nodes.append(node)
            opset_imports = [make_opsetid(domain, 1 if version is None else version)
                             for domain, version in opsets.items()]

            graph = make_graph(
                nodes, 'test_bernoulli', inputs, outputs, initializers)

            onnx_model = make_model(
                graph, opset_imports=opset_imports, functions=functions)
            onnx_model.ir_version = 8
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})
            return onnx_model

        onnx_model = create_model()

        oinf = OnnxInference(onnx_model)
        xs = [array([0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548,
                     0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152])]
        ys = [array([0., 1., 1., 0., 0., 1., 0., 1., 1., 1.])]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqual(y.dtype, gy.dtype)
            self.assertEqual(y.shape, gy.shape)

    def test_enumerate_onnx_tests_test_bernoulli_cpu(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_bernoulli'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_constantofshape_int_shape_zero_code(self):
        name = 'test_constantofshape_int_shape_zero'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn(
            'def test_constantofshape_int_shape_zero(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #     print(code[0])

    def test_constantofshape_int_shape_zero2(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []
            functions = []

            opsets = {'': 12}
            inputs.append(make_tensor_value_info('x', 7, [1]))
            outputs.append(make_tensor_value_info('y', 6, [None]))
            node = make_node(
                'ConstantOfShape', ['x'], ['y'],
                value=make_tensor(
                    "value", TensorProto.INT32, dims=[1], vals=[0]), domain='')
            nodes.append(node)

            opset_imports = [make_opsetid(domain, 1 if version is None else version)
                             for domain, version in opsets.items()]
            graph = make_graph(
                nodes, 'test_constantofshape_int_shape_zero', inputs, outputs, initializers)
            onnx_model = make_model(
                graph, opset_imports=opset_imports, functions=functions)
            onnx_model.ir_version = 6
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})
            return onnx_model

        onnx_model = create_model()
        oinf = OnnxInference(onnx_model)
        xs = [array([0], dtype=int64)]
        ys = [array([], dtype=int32)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            self.assertEqualArray(y, gy)

    def test_enumerate_onnx_test_constantofshape_int_shape_zero_code(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_constantofshape_int_shape_zero'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_cumsum_1d_exclusive(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_cumsum_1d_exclusive'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_min_example(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_min_example'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_eyelike_without_dtype(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_eyelike_without_dtype'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_sce_mean_expanded(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_sce_mean_expanded'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_dynamicquantizelinear(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_dynamicquantizelinear'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_isinf_negative(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_isinf_negative'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_selu(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_selu'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_sce_mean_weight_expanded(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_sce_mean_weight_expanded'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_shape_end(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_shape_end_1'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_nonzero_example(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_nonzero_example'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_mod_mixed_sign_float16(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_mod_mixed_sign_float16'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_max_one_input(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_max_one_input'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_eyelike_without_dtype_2(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_eyelike_without_dtype'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_dynamicquantizelinear_max_adjusted_expanded_code(self):
        name = 'test_dynamicquantizelinear_max_adjusted_expanded'
        code = []
        for te in enumerate_onnx_tests('node', lambda folder: folder == name):
            code.append(te.to_python())
        self.assertEqual(len(code), 1)
        self.assertIn(
            'def test_dynamicquantizelinear_max_adjusted_expanded(self):', code[0])
        self.assertIn('from onnx.helper', code[0])
        self.assertIn('for y, gy in zip(ys, goty):', code[0])
        # if __name__ == '__main__':
        #     print(code[0])

    def test_dynamicquantizelinear_max_adjusted_expanded(self):

        def create_model():
            initializers = []
            nodes = []
            inputs = []
            outputs = []
            functions = []

            opsets = {'': 11}
            inputs.append(make_tensor_value_info('x', 1, [6]))
            outputs.append(make_tensor_value_info('y', 2, [6]))
            outputs.append(make_tensor_value_info('y_scale', 1, None))
            outputs.append(make_tensor_value_info('y_zero_point', 2, None))
            node = make_node(
                'Constant', [], ['var__functionQ_Min'],
                value=make_tensor("value", TensorProto.FLOAT, dims=[], vals=[0.0]), domain='')
            nodes.append(node)

            node = make_node(
                'Constant', [], ['var__functionQ_Max'],
                value=make_tensor("value", TensorProto.FLOAT, dims=[], vals=[255.0]), domain='')
            nodes.append(node)

            node = make_node(
                'ReduceMin', ['x'], ['var__functionX_Min'], keepdims=0, domain='')
            nodes.append(node)

            node = make_node(
                'Min', ['var__functionX_Min', 'var__functionQ_Min'],
                ['var__functionX_Min_Adjusted'], domain='')
            nodes.append(node)

            node = make_node(
                'ReduceMax', ['x'], ['var__functionX_Max'], keepdims=0, domain='')
            nodes.append(node)

            node = make_node(
                'Max', ['var__functionX_Max', 'var__functionQ_Min'],
                ['var__functionX_Max_Adjusted'], domain='')
            nodes.append(node)

            node = make_node(
                'Sub', ['var__functionX_Max_Adjusted',
                        'var__functionX_Min_Adjusted'],
                ['var__functionX_Range'], domain='')
            nodes.append(node)

            node = make_node(
                'Div', ['var__functionX_Range', 'var__functionQ_Max'],
                ['var__functionScale'], domain='')
            nodes.append(node)

            node = make_node(
                'Div', ['var__functionX_Min_Adjusted', 'var__functionScale'],
                ['var__functionMin_Scaled'], domain='')
            nodes.append(node)

            node = make_node(
                'Sub', ['var__functionQ_Min', 'var__functionMin_Scaled'],
                ['var__functionInitial_ZeroPoint_FP'], domain='')
            nodes.append(node)

            node = make_node(
                'Clip', ['var__functionInitial_ZeroPoint_FP',
                         'var__functionQ_Min', 'var__functionQ_Max'],
                ['var__functionClipped_ZeroPoint_FP'], domain='')
            nodes.append(node)

            node = make_node(
                'Round', ['var__functionClipped_ZeroPoint_FP'],
                ['var__functionRounded_ZeroPoint_FP'], domain='')
            nodes.append(node)

            node = make_node(
                'Cast', ['var__functionRounded_ZeroPoint_FP'],
                ['var__functionZeropoint'], to=TensorProto.UINT8, domain='')
            nodes.append(node)

            node = make_node(
                'Identity', ['var__functionScale'], ['y_scale'], domain='')
            nodes.append(node)

            node = make_node(
                'Identity', ['var__functionZeropoint'], ['y_zero_point'], domain='')
            nodes.append(node)

            node = make_node(
                'QuantizeLinear', [
                    'x', 'var__functionScale', 'var__functionZeropoint'],
                ['y'], domain='')
            nodes.append(node)

            opset_imports = [make_opsetid(domain, 1 if version is None else version)
                             for domain, version in opsets.items()]

            graph = make_graph(
                nodes, 'test_dynamicquantizelinear_max_adjusted_expanded', inputs, outputs, initializers)

            onnx_model = make_model(
                graph, opset_imports=opset_imports, functions=functions)
            onnx_model.ir_version = 5
            onnx_model.producer_name = 'backend-test'
            onnx_model.producer_version = ''
            onnx_model.domain = ''
            onnx_model.model_version = 0
            onnx_model.doc_string = ''
            set_model_props(onnx_model, {})

            return onnx_model

        onnx_model = create_model()

        oinf = OnnxInference(onnx_model)
        xs = [array([-1., -2.1, -1.3, -2.5, -3.34, -4.], dtype=float32)]
        ys = [array([191, 121, 172, 96, 42, 0], dtype=uint8),
              array(0.01568628, dtype=float32),
              array(255, dtype=uint8)]
        feeds = {n: x for n, x in zip(oinf.input_names, xs)}
        got = oinf.run(feeds)
        goty = [got[k] for k in oinf.output_names]
        for y, gy in zip(ys, goty):
            diff = numpy.abs(y - gy).sum()
            self.assertLess(diff, 2)

    def test_enumerate_onnx_test_range_float_type_positive_delta_expanded(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_range_float_type_positive_delta_expanded'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    @unittest.skipIf(True, reason="onnx example is probably wrong")
    def test_enumerate_onnx_test_simple_rnn_batchwise(self):
        # The test may fail but the numerical result may be different
        # depending on the machine.
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_simple_rnn_batchwise'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct,
                   decimal=2)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_blackman_window(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_blackmanwindow'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_hann_window(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_hannwindow'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_hamming_window(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_hammingwindow'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_dft(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_dft'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_dft_axis(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_dft_axis'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_dft_inverse(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_dft_inverse'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_layer_normalization_2d_axis0(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_layer_normalization_2d_axis0'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    @unittest.skipIf(True, reason="unfinished")
    def test_enumerate_onnx_test_stft(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_stft'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_tril_neg(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_tril_neg'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_tril_zero(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_tril_zero'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def test_enumerate_onnx_test_triu_neg(self):
        done = 0
        for te in enumerate_onnx_tests(
                'node', lambda folder: folder == 'test_triu_neg'):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)


if __name__ == "__main__":
    # TestOnnxBackEnd().test_enumerate_onnx_tests_run_one()
    unittest.main()
