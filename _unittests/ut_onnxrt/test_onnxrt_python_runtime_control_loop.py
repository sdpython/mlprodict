"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from onnx.helper import (
    make_tensor_value_info, make_node, make_graph,
    make_operatorsetid, make_sequence_value_info,
    make_tensor, make_model)
from onnx import TensorProto
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.type_object import SequenceType
from mlprodict.tools import get_opset_number_from_onnx


def make_tensor_sequence_value_info(name, tensor_type, shape):
    return make_sequence_value_info(
        name, tensor_type, shape, None)


class TestOnnxrtPythonRuntimeControlLoop(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(DeprecationWarning)
    def test_sequence_insert(self):

        def expect(node, inputs, outputs, name):
            ginputs = [
                make_sequence_value_info(
                    node.input[0], TensorProto.FLOAT, []),  # pylint: disable=E1101,
                make_sequence_value_info(
                    node.input[1], TensorProto.FLOAT, []),  # pylint: disable=E1101,
            ]
            if len(node.input) > 2:
                ginputs.append(
                    make_tensor_value_info(
                        node.input[2], TensorProto.INT64, []),  # pylint: disable=E1101
                )
            goutputs = [
                make_sequence_value_info(
                    node.output[0], TensorProto.FLOAT, []),  # pylint: disable=E1101,
            ]
            model_def = make_model(
                opset_imports=[
                    make_operatorsetid('', get_opset_number_from_onnx())],
                graph=make_graph(
                    name=name, inputs=ginputs, outputs=goutputs,
                    nodes=[node]))
            oinf = OnnxInference(model_def)
            got = oinf.run({n: v for n, v in zip(node.input, inputs)})
            self.assertEqual(len(got), 1)
            oseq = got['output_sequence']
            self.assertEqual(len(oseq), len(outputs))
            for e, g in zip(outputs, oseq):
                self.assertEqualArray(e, g)

        test_cases = {
            'at_back': [numpy.array([10, 11, 12]).astype(numpy.int64)],
            'at_front': [numpy.array([-2, -1, 0]),
                         numpy.array([0]).astype(numpy.int64)]}
        sequence = [numpy.array([1, 2, 3, 4]).astype(numpy.int64),
                    numpy.array([5, 6, 7]).astype(numpy.int64),
                    numpy.array([8, 9]).astype(numpy.int64)]

        for test_name, test_inputs in test_cases.items():
            with self.subTest(test_name=test_name):
                tensor = test_inputs[0].astype(numpy.int64)

                if len(test_inputs) > 1:
                    node = make_node(
                        'SequenceInsert',
                        inputs=['sequence', 'tensor', 'position'],
                        outputs=['output_sequence'])
                    position = test_inputs[1]
                    inserted = self.sequence_insert_reference_implementation(
                        sequence, tensor, position)
                    expect(node, inputs=[sequence, tensor, position], outputs=inserted,
                           name='test_sequence_insert_' + test_name)
                else:
                    node = make_node(
                        'SequenceInsert',
                        inputs=['sequence', 'tensor'],
                        outputs=['output_sequence'])
                    inserted = self.sequence_insert_reference_implementation(
                        sequence, tensor)
                    expect(node, inputs=[sequence, tensor], outputs=inserted,
                           name='test_sequence_insert_' + test_name)

    @ignore_warnings(DeprecationWarning)
    def test_loop(self):
        # Given a tensor x of values [x1, ..., xN],
        # Return a sequence of tensors of
        #   [[x1], [x1, x2], ..., [x1, ..., xN]]

        cond_in = make_tensor_value_info(
            'cond_in', TensorProto.BOOL, [])  # pylint: disable=E1101
        cond_out = make_tensor_value_info(
            'cond_out', TensorProto.BOOL, [])  # pylint: disable=E1101
        iter_count = make_tensor_value_info(
            'iter_count', TensorProto.INT64, [])  # pylint: disable=E1101
        seq_in = make_tensor_sequence_value_info(
            'seq_in', TensorProto.FLOAT, None)  # pylint: disable=E1101
        seq_out = make_tensor_sequence_value_info(
            'seq_out', TensorProto.FLOAT, None)  # pylint: disable=E1101

        x = numpy.array([1, 2, 3, 4, 5]).astype(numpy.float32)

        x_const_node = make_node(
            'Constant', inputs=[], outputs=['x'],
            value=make_tensor(
                name='const_tensor_x', data_type=TensorProto.FLOAT,  # pylint: disable=E1101
                dims=x.shape, vals=x.flatten().astype(float)))

        one_const_node = make_node(
            'Constant', inputs=[], outputs=['one'],
            value=make_tensor(
                name='const_tensor_one', data_type=TensorProto.INT64,  # pylint: disable=E1101
                dims=(), vals=[1]))

        zero_const_node = make_node(
            'Constant', inputs=[], outputs=['slice_start'],
            value=make_tensor(
                name='const_tensor_zero', data_type=TensorProto.INT64,  # pylint: disable=E1101
                dims=(1,), vals=[0]))

        axes_node = make_node(
            'Constant', inputs=[], outputs=['axes'],
            value=make_tensor(
                name='const_tensor_axes', data_type=TensorProto.INT64,  # pylint: disable=E1101
                dims=(), vals=[0]))

        add_node = make_node(
            'Add', inputs=['iter_count', 'one'], outputs=['end'])

        end_unsqueeze_node = make_node(
            'Unsqueeze', inputs=['end', 'axes'], outputs=['slice_end'])

        slice_node = make_node(
            'Slice', inputs=['x', 'slice_start', 'slice_end'], outputs=['slice_out'])

        insert_node = make_node(
            'SequenceInsert', inputs=['seq_in', 'slice_out'], outputs=['seq_out'])

        identity_node = make_node(
            'Identity', inputs=['cond_in'], outputs=['cond_out'])

        loop_body = make_graph(
            [identity_node, x_const_node, one_const_node, zero_const_node, add_node,
             axes_node, end_unsqueeze_node, slice_node, insert_node],
            'loop_body', [iter_count, cond_in, seq_in], [cond_out, seq_out])

        node = make_node(
            'Loop', inputs=['trip_count', 'cond', 'seq_empty'],
            outputs=['seq_res'], body=loop_body)
        node_concat = make_node(
            'ConcatFromSequence', inputs=['seq_res'],
            outputs=['res'], axis=0, new_axis=0)

        trip_count = numpy.array(5).astype(numpy.int64)
        seq_empty = []  # type: List[Any]
        # seq_res = [x[:int(i)] for i in x]
        cond = numpy.array(1).astype(numpy.bool)

        model_def = make_model(
            opset_imports=[
                make_operatorsetid('', get_opset_number_from_onnx())],
            graph=make_graph(
                name='loop_test',
                inputs=[
                    make_tensor_value_info(
                        'trip_count', TensorProto.INT64, trip_count.shape),  # pylint: disable=E1101
                    make_tensor_value_info(
                        'cond', TensorProto.BOOL, cond.shape),  # pylint: disable=E1101
                    make_sequence_value_info(
                        'seq_empty', TensorProto.FLOAT, [])],  # pylint: disable=E1101
                outputs=[make_tensor_value_info(
                    'res', TensorProto.FLOAT, None)],  # pylint: disable=E1101
                nodes=[node, node_concat]))

        expected = numpy.array([
            1., 1., 2., 1., 2., 3., 1., 2.,
            3., 4., 1., 2., 3., 4., 5.], dtype=numpy.float32)
        for rt in ['onnxruntime1', 'python', 'python_compiled']:
            with self.subTest(rt=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                inputs = {
                    'trip_count': trip_count, 'cond': cond,
                    'seq_empty': seq_empty}
                got = oinf.run(inputs)
                self.assertEqualArray(expected, got['res'])
                if rt == 'python':
                    siz = oinf.infer_sizes(inputs)
                    self.assertIsInstance(siz, dict)
                    typ = oinf.infer_types()
                    self.assertEqual(typ["trip_count"], numpy.int64)
                    if 'cond' in typ:
                        self.assertEqual(typ["cond"], numpy.bool_)
                    for k, v in typ.items():
                        if k in {'trip_count', 'cond'}:
                            continue
                        self.assertIsInstance(v, SequenceType)

    @ignore_warnings(DeprecationWarning)
    def test_loop_additional_input(self):
        # Given a tensor x of values [x1, ..., xN],
        # Return a sequence of tensors of
        #   [[x1], [x1, x2], ..., [x1, ..., xN]]

        cond_in = make_tensor_value_info(
            'cond_in', TensorProto.BOOL, [])  # pylint: disable=E1101
        cond_out = make_tensor_value_info(
            'cond_out', TensorProto.BOOL, [])  # pylint: disable=E1101
        iter_count = make_tensor_value_info(
            'iter_count', TensorProto.INT64, [])  # pylint: disable=E1101
        seq_in = make_tensor_sequence_value_info(
            'seq_in', TensorProto.FLOAT, None)  # pylint: disable=E1101
        seq_out = make_tensor_sequence_value_info(
            'seq_out', TensorProto.FLOAT, None)  # pylint: disable=E1101

        x = numpy.array([1, 2, 3, 4, 5]).astype(numpy.float32)

        x_const_node = make_node(
            'Constant', inputs=[], outputs=['x'],
            value=make_tensor(
                name='const_tensor_x', data_type=TensorProto.FLOAT,  # pylint: disable=E1101
                dims=x.shape, vals=x.flatten().astype(float)))

        zero_const_node = make_node(
            'Constant', inputs=[], outputs=['slice_start'],
            value=make_tensor(
                name='const_tensor_zero', data_type=TensorProto.INT64,  # pylint: disable=E1101
                dims=(1,), vals=[0]))

        axes_node = make_node(
            'Constant', inputs=[], outputs=['axes'],
            value=make_tensor(
                name='const_tensor_axes', data_type=TensorProto.INT64,  # pylint: disable=E1101
                dims=(), vals=[0]))

        add_node = make_node(
            'Add', inputs=['iter_count', 'XI'], outputs=['slice_end'])

        slice_node = make_node(
            'Slice', inputs=['x', 'slice_start', 'slice_end'], outputs=['slice_out'])

        insert_node = make_node(
            'SequenceInsert', inputs=['seq_in', 'slice_out'], outputs=['seq_out'])

        identity_node = make_node(
            'Identity', inputs=['cond_in'], outputs=['cond_out'])

        loop_body = make_graph(
            [identity_node, x_const_node, zero_const_node, add_node,
             axes_node, slice_node, insert_node],
            'loop_body', [iter_count, cond_in, seq_in], [cond_out, seq_out])

        node = make_node(
            'Loop', inputs=['trip_count', 'cond', 'seq_empty'],
            outputs=['seq_res'], body=loop_body)
        node1 = make_node('Neg', inputs=['XI'], outputs=['Y'])
        node_concat = make_node(
            'ConcatFromSequence', inputs=['seq_res'],
            outputs=['res'], axis=0, new_axis=0)

        trip_count = numpy.array(5).astype(numpy.int64)
        seq_empty = []  # type: List[Any]
        cond = numpy.array(1).astype(numpy.bool)

        model_def = make_model(
            opset_imports=[
                make_operatorsetid('', get_opset_number_from_onnx())],
            graph=make_graph(
                name='loop_test',
                inputs=[
                    make_tensor_value_info(
                        'trip_count', TensorProto.INT64, trip_count.shape),  # pylint: disable=E1101
                    make_tensor_value_info(
                        'cond', TensorProto.BOOL, cond.shape),  # pylint: disable=E1101
                    make_sequence_value_info(
                        'seq_empty', TensorProto.FLOAT, []),  # pylint: disable=E1101
                    make_tensor_value_info(
                        'XI', TensorProto.INT64, [])],  # pylint: disable=E1101
                outputs=[
                    make_tensor_value_info(
                        'res', TensorProto.FLOAT, None),  # pylint: disable=E1101
                    make_tensor_value_info(
                        'Y', TensorProto.INT64, [])],  # pylint: disable=E1101
                nodes=[node1, node, node_concat]))

        del model_def.opset_import[:]  # pylint: disable=E1101
        op_set = model_def.opset_import.add()
        op_set.domain = ''
        op_set.version = 14
        model_def.ir_version = 7

        expected = numpy.array([
            1., 1., 2., 1., 2., 3., 1., 2.,
            3., 4., 1., 2., 3., 4., 5.], dtype=numpy.float32)
        X = numpy.array([1], dtype=numpy.int64)
        for rt in ['python', 'onnxruntime1', 'python_compiled']:
            with self.subTest(rt=rt):
                oinf = OnnxInference(model_def, runtime=rt)
                inputs = {
                    'trip_count': trip_count, 'cond': cond,
                    'seq_empty': seq_empty,
                    'XI': X}
                if rt == 'python_compiled':
                    code = str(oinf)
                    self.assertIn("context={'XI': XI}", code)
                got = oinf.run(inputs)
                self.assertEqualArray(-X, got['Y'])
                self.assertEqualArray(expected, got['res'])
                if rt == 'python':
                    siz = oinf.infer_sizes(inputs)
                    self.assertIsInstance(siz, dict)
                    typ = oinf.infer_types()
                    self.assertEqual(typ["trip_count"], numpy.int64)
                    if 'cond' in typ:
                        self.assertEqual(typ["cond"], numpy.bool_)
                    for k, v in typ.items():
                        if k in {'trip_count', 'cond', 'Y', 'XI'}:
                            continue
                        self.assertIsInstance(v, SequenceType)

    def sequence_insert_reference_implementation(
            self, sequence, tensor, position=None):
        seq = list(sequence)
        if position is not None:
            insert_position = position[0]
            seq.insert(insert_position, tensor)
        else:
            seq.append(tensor)
        return seq


if __name__ == "__main__":
    unittest.main()
