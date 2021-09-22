"""
@brief      test log(time=2s)
"""
import unittest
from collections import OrderedDict
import numpy
from onnx import helper, TensorProto
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxMul, OnnxSub, OnnxIdentity, OnnxScan,
    OnnxReduceSumSquare, OnnxSqueezeApi11)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnx_tools.optim.onnx_helper import onnx_statistics
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.optim import onnx_remove_node_unused
from mlprodict.onnx_tools.onnx_manipulations import (
    select_model_inputs_outputs, enumerate_model_node_outputs,
    onnx_rename_names, insert_results_into_onnx)
from mlprodict.tools import get_opset_number_from_onnx


class TestOptimOnnxManipulations(ExtTestCase):

    def test_onnx_remove_unused_outputs(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=get_opset_number_from_onnx()),
            cop2, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def, "inter", infer_shapes=True, remove_unused=False)
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_unused(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 2)
        self.assertEqual(stats2['ninits'], 2)
        self.assertEqual(stats3['ninits'], 1)
        self.assertEqual(stats2['nnodes'], 1)
        self.assertEqual(stats3['nnodes'], 1)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'X': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'X': x})
        self.assertNotIn('final', y1)
        self.assertNotIn('final', y2)
        self.assertIn('inter', y1)
        self.assertIn('inter', y2)
        self.assertEqualArray(y1['inter'], y2['inter'])

    def test_onnx_remove_unused_outputs_new(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=get_opset_number_from_onnx()),
            cop2, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def0 = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def0, "inter", infer_shapes=True, remove_unused=False)
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = select_model_inputs_outputs(
            model_def0, "inter", infer_shapes=True)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 2)
        self.assertEqual(stats2['ninits'], 2)
        self.assertEqual(stats3['ninits'], 1)
        self.assertEqual(stats2['nnodes'], 1)
        self.assertEqual(stats3['nnodes'], 1)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'X': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'X': x})
        self.assertNotIn('final', y1)
        self.assertNotIn('final', y2)
        self.assertIn('inter', y1)
        self.assertIn('inter', y2)
        self.assertEqualArray(y1['inter'], y2['inter'])

    def test_onnx_remove_unused_inputs(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', cop2,
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop3, cop3, op_version=get_opset_number_from_onnx()),
            cop3, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def, inputs=["inter"], infer_shapes=True, remove_unused=False)
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_unused(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 1)
        self.assertEqual(stats2['ninits'], 1)
        self.assertEqual(stats3['ninits'], 0)
        self.assertEqual(stats2['nnodes'], 2)
        self.assertEqual(stats3['nnodes'], 2)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'inter': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'inter': x})
        self.assertIn('final', y1)
        self.assertIn('final', y2)
        self.assertNotIn('inter', y1)
        self.assertNotIn('inter', y2)
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_remove_unused_inputs_overwrite(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', cop2,
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop3, cop3, op_version=get_opset_number_from_onnx()),
            cop3, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        model_def = select_model_inputs_outputs(
            model_def, inputs=["inter"], infer_shapes=False,
            overwrite=dict(inter=(numpy.float32, [None, None]),
                           final=(numpy.float32, [None, None])),
            remove_unused=False)
        stats = onnx_statistics(model_def, optim=True)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_unused(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def, optim=True)
        stats3 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['ninits'], 1)
        self.assertEqual(stats2['ninits'], 1)
        self.assertEqual(stats3['ninits'], 0)
        self.assertEqual(stats2['nnodes'], 2)
        self.assertEqual(stats3['nnodes'], 2)
        oinf1 = OnnxInference(model_def)
        y1 = oinf1.run({'inter': x})

        oinf2 = OnnxInference(new_model)
        y2 = oinf2.run({'inter': x})
        self.assertIn('final', y1)
        self.assertIn('final', y2)
        self.assertNotIn('inter', y1)
        self.assertNotIn('inter', y2)
        self.assertEqualArray(y1['final'], y2['final'])

    def test_enumerate_model_node_outputs(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=get_opset_number_from_onnx()),
            cop2, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        nodes1 = list(enumerate_model_node_outputs(model_def))
        nodes2 = list(enumerate_model_node_outputs(model_def, order=True))
        self.assertEqual(list(sorted(nodes1)), list(sorted(nodes2)))
        expected = ['Ad_Addcst2', 'Ad_C0', 'inter', 'Ad_C02', 'Mu_C0', 'final']
        self.assertEqual(nodes2, expected)

    def test_onnx_rename_names_exc(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=get_opset_number_from_onnx()),
            cop2, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        self.assertRaise(
            lambda: onnx_rename_names(model_def, strategy="none"),
            ValueError)

    def test_onnx_rename_names_simple(self):
        rows = []

        def flog(*s):
            rows.append(" ".join(map(str, s)))

        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=get_opset_number_from_onnx()),
            cop2, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        oinf1 = OnnxInference(model_def)
        new_model = onnx_rename_names(model_def, verbose=1, fLOG=flog)
        total = "\n".join(rows)
        self.assertIn("[onnx_rename_names] 'Ad_Addcst1' -> 'i1'", total)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'X': x})
        y2 = oinf2.run({'X': x})
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_rename_names_type(self):
        rows = []

        def flog(*s):
            rows.append(" ".join(map(str, s)))

        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=get_opset_number_from_onnx())
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=get_opset_number_from_onnx())
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=get_opset_number_from_onnx(),
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=get_opset_number_from_onnx()),
            cop2, output_names=['final'],
            op_version=get_opset_number_from_onnx())
        model_def = cop4.to_onnx({'X': x})
        oinf1 = OnnxInference(model_def)
        new_model = onnx_rename_names(
            model_def, verbose=1, fLOG=flog, strategy='type')
        total = "\n".join(rows)
        self.assertIn("'Ad_Addcst' -> 'i_05'", total)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'X': x})
        y2 = oinf2.run({'X': x})
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_rename_node_scan(self):

        def squareform_pdist(X, **kwargs):
            opv = get_opset_number_from_onnx()
            diff = OnnxSub('next_in', 'next', output_names=[
                           'diff'], op_version=opv)
            id_next = OnnxIdentity('next_in', output_names=[
                                   'next_out'], op_version=opv)
            norm = OnnxReduceSumSquare(
                diff, output_names=['norm'], axes=[1], op_version=opv)
            flat = OnnxSqueezeApi11(
                norm, output_names=['scan_out'], axes=[1], op_version=opv)
            scan_body = id_next.to_onnx(
                OrderedDict([('next_in', FloatTensorType()),
                             ('next', FloatTensorType())]),
                outputs=[('next_out', FloatTensorType([None, None])),
                         ('scan_out', FloatTensorType([None]))],
                other_outputs=[flat])

            node = OnnxScan(X, X, output_names=['scan0_{idself}', 'scan1_{idself}'],
                            num_scan_inputs=1, body=scan_body.graph, op_version=opv,
                            **kwargs)
            return node[1]

        rows = []

        def flog(*s):
            rows.append(" ".join(map(str, s)))

        opv = get_opset_number_from_onnx()
        onnx_fct = OnnxIdentity(squareform_pdist(
            'x'), output_names='Y', op_version=opv)
        model_def = onnx_fct.to_onnx(inputs=[('x', FloatTensorType())])

        oinf1 = OnnxInference(model_def)
        new_model = onnx_rename_names(
            model_def, verbose=1, fLOG=flog, strategy='type')
        total = "\n".join(rows)
        self.assertNotIn('name: "Re_ReduceSumSquare"', str(new_model))
        self.assertIn("'Re_ReduceSumSquare' -> 'n_24'", total)
        oinf2 = OnnxInference(new_model)
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        y1 = oinf1.run({'x': x})
        y2 = oinf2.run({'x': x})
        self.assertEqualArray(y1['Y'], y2['Y'])

    def test_insert_results_into_onnx(self):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, None)  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.INT64, None)  # pylint: disable=E1101
        node_def = helper.make_node('Shape', ['X'], ['Z0'], name='Zt')
        node_def1 = helper.make_node('Identity', ['Z0'], ['Z'], name='Zti')
        graph_def = helper.make_graph(
            [node_def, node_def1], 'test-model', [X], [Z])
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict',
            ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 13)])

        new_graph = insert_results_into_onnx(
            model_def, {'Z0': numpy.array([[29, 39]], dtype=numpy.int64)})
        s_graph = str(new_graph)
        self.assertIn('domain: "DEBUG"', s_graph)
        self.assertNotIn('pname', s_graph)
        self.assertIn('op_type: "DEBUG"', s_graph)
        self.assertRaise(lambda: insert_results_into_onnx(
            model_def, {'Zt': numpy.array([29, 39], dtype=numpy.int64)}),
            RuntimeError)
        # with open('debug.onnx', 'wb') as f:
        #     f.write(new_graph.SerializeToString())

        oinf1 = OnnxInference(model_def, inplace=False)
        oinf2 = OnnxInference(new_graph, inplace=False)
        cst = numpy.array([[5.6, 7.8]])
        self.assertEqualArray(oinf1.run({'X': cst})['Z'],
                              oinf2.run({'X': cst})['Z'])

        onx = oinf1.run2onnx({'X': cst})[1]
        s_graph = str(onx)
        self.assertIn('domain: "DEBUG"', s_graph)
        self.assertIn('op_type: "DEBUG"', s_graph)
        self.assertNotIn('pname', s_graph)
        oinf3 = OnnxInference(onx)
        self.assertEqualArray(oinf1.run({'X': cst})['Z'],
                              oinf3.run({'X': cst})['Z'])

    def test_insert_results_into_onnx_init(self):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, None)  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.INT64, None)  # pylint: disable=E1101
        node_def = helper.make_node('Shape', ['X'], ['Z0'], name='Zt')
        node_def1 = helper.make_node('Identity', ['Z0'], ['Z'], name='Zti')
        graph_def = helper.make_graph(
            [node_def, node_def1], 'test-model', [X], [Z])
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict',
            ir_version=7, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', 13)])

        new_graph = insert_results_into_onnx(
            model_def, {'Z0': numpy.array([[29, 39]], dtype=numpy.int64)},
            as_parameter=False, param_name=lambda k: k)
        s_graph = str(new_graph)
        self.assertIn('domain: "DEBUG"', s_graph)
        self.assertIn('op_type: "DEBUG"', s_graph)
        self.assertRaise(lambda: insert_results_into_onnx(
            model_def, {'Zt': numpy.array([29, 39], dtype=numpy.int64)}),
            RuntimeError)
        self.assertRaise(lambda: insert_results_into_onnx(
            model_def, {'X': numpy.array([29, 39], dtype=numpy.int64)}),
            NotImplementedError)
        # with open('debug.onnx', 'wb') as f:
        #     f.write(new_graph.SerializeToString())

        oinf1 = OnnxInference(model_def)
        oinf2 = OnnxInference(new_graph)
        cst = numpy.array([[5.6, 7.8]])
        self.assertEqualArray(oinf1.run({'X': cst})['Z'],
                              oinf2.run({'X': cst})['Z'])


if __name__ == "__main__":
    unittest.main()
