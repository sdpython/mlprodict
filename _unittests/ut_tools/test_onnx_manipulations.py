"""
@brief      test log(time=2s)
"""
import unittest
import os
import numpy
from onnx import helper, TensorProto, load, FunctionProto
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xop import loadop, OnnxOperatorFunction
from mlprodict.npy.xop_variable import Variable
from mlprodict.onnx_tools.optim.onnx_helper import onnx_statistics
from mlprodict.onnx_tools.onnx_tools import enumerate_onnx_names
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.optim import onnx_remove_node_unused
from mlprodict.onnx_tools.onnx_manipulations import (
    select_model_inputs_outputs, enumerate_model_node_outputs,
    onnx_rename_names, insert_results_into_onnx, onnx_model_to_function,
    onnx_inline_function)
from mlprodict import __max_supported_opset__ as TARGET_OPSET


class TestOptimOnnxManipulations(ExtTestCase):

    def test_onnx_remove_unused_outputs(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
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
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
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
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', cop2,
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop3, cop3, op_version=TARGET_OPSET),
            cop3, output_names=['final'],
            op_version=TARGET_OPSET)
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
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', cop2,
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop3, cop3, op_version=TARGET_OPSET),
            cop3, output_names=['final'],
            op_version=TARGET_OPSET)
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
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        nodes1 = list(enumerate_model_node_outputs(model_def))
        nodes2 = list(enumerate_model_node_outputs(model_def, order=True))
        self.assertEqual(list(sorted(nodes1)), list(sorted(nodes2)))
        expected = ['inter', 'out_add_0', 'out_mul_0', 'final']
        self.assertEqual(nodes2, expected)

    def test_onnx_rename_names_exc(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        self.assertRaise(
            lambda: onnx_rename_names(model_def, strategy="none"),
            ValueError)

    def test_onnx_rename_names_simple(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        rows = []

        def flog(*s):
            rows.append(" ".join(map(str, s)))

        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        oinf1 = OnnxInference(model_def)
        new_model = onnx_rename_names(model_def, verbose=1, fLOG=flog)
        total = "\n".join(rows)
        self.assertIn("[onnx_rename_names] init: 'init_1' -> 'i1'", total)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'X': x})
        y2 = oinf2.run({'X': x})
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_rename_names_type(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        rows = []

        def flog(*s):
            rows.append(" ".join(map(str, s)))

        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        oinf1 = OnnxInference(model_def)
        new_model = onnx_rename_names(
            model_def, verbose=1, fLOG=flog, strategy='type')
        total = "\n".join(rows)
        self.assertIn("'init' -> 'i_DB'", total)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'X': x})
        y2 = oinf2.run({'X': x})
        self.assertEqualArray(y1['final'], y2['final'])

    def test_onnx_rename_node_scan(self):
        (OnnxSub, OnnxReduceSumSquare,
         OnnxIdentity, OnnxScan) = loadop(
            'Sub', 'ReduceSumSquare', 'Identity', 'Scan')

        def onnx_squareform_pdist(X, dtype=None, op_version=None, **kwargs):
            diff = OnnxSub('next_in', 'next',
                           op_version=op_version)
            id_next = OnnxIdentity('next_in', output_names=['next_out'],
                                   op_version=op_version)
            flat = OnnxReduceSumSquare(diff, axes=[1], op_version=op_version,
                                       output_names=['scan_out'], keepdims=0)
            scan_body = id_next.to_onnx(
                [Variable('next_in', numpy.float32, (None, None)),  # tensor_type([None, None])),
                 Variable('next', numpy.float32, (None, ))],  # tensor_type([None]))]),
                outputs=[Variable('next_out', numpy.float32, (None, None)),  # ([None, None])),
                         Variable('scan_out', numpy.float32, (None, ))],  # tensor_type([None]))],
                other_outputs=[flat],
                target_opset=op_version)
            node = OnnxScan(X, X, output_names=['S1', 'S2'],
                            num_scan_inputs=1,
                            body=(scan_body.graph, [id_next, flat]),
                            op_version=op_version, **kwargs)
            return node[1]

        rows = []

        def flog(*s):
            rows.append(" ".join(map(str, s)))

        opv = TARGET_OPSET
        onnx_fct = OnnxIdentity(onnx_squareform_pdist(
            'x'), output_names='Y', op_version=opv)
        model_def = onnx_fct.to_onnx(inputs={'x': numpy.float32})

        oinf1 = OnnxInference(model_def)
        new_model = onnx_rename_names(
            model_def, verbose=1, fLOG=flog, strategy='type')
        total = "\n".join(rows)
        self.assertNotIn('name: "Re_ReduceSumSquare"', str(new_model))
        self.assertIn("'node__reducesumsquare_", total)
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

    def test_onnx_enumerate_onnx_names(self):
        OnnxAdd, OnnxSub, OnnxMul = loadop('Add', 'Sub', 'Mul')
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype),
                      op_version=TARGET_OPSET)
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype),
                       op_version=TARGET_OPSET)
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype),
                       op_version=TARGET_OPSET,
                       output_names=['inter'])
        cop4 = OnnxSub(
            OnnxMul(cop, cop3, op_version=TARGET_OPSET),
            cop2, output_names=['final'],
            op_version=TARGET_OPSET)
        model_def = cop4.to_onnx({'X': x})
        names = list(enumerate_onnx_names(model_def))
        self.assertEqual(len(names), 16)
        self.assertIn('X', names)
        self.assertIn('inter', names)

    def test_onnx_to_function(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        fft2d = os.path.join(data, "fft2d.onnx")
        onx = load(fft2d)

        # original graph
        oinf = OnnxInference(onx)
        x = numpy.random.randn(7, 7).astype(numpy.float32)
        y = oinf.run({'x': x})['y']

        fct = onnx_model_to_function(onx, name="fft2d")
        self.assertIsInstance(fct, FunctionProto)

        op = OnnxOperatorFunction(fct, 'X', output_names=['Y'])
        onx2 = op.to_onnx(numpy.float32, numpy.float32)
        s2 = str(onx2)
        self.assertIn("functions {", s2)
        self.assertIn('name: "fft2d"', s2)
        oinf2 = OnnxInference(onx2)
        y2 = oinf2.run({'X': x})['Y']
        self.assertEqualArray(y, y2)

    def test_onnx_inline_function(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        fft2d = os.path.join(data, "fft2d.onnx")
        onx = load(fft2d)
        fct = onnx_model_to_function(onx, name="fft2d")
        op = OnnxOperatorFunction(fct, 'X', output_names=['Y'])
        onx2 = op.to_onnx(numpy.float32, numpy.float32)
        inlined, m = onnx_inline_function(onx2)
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].op_type, "fft2d")
        s3 = str(inlined)
        self.assertNotIn("functions {", s3)

        x = numpy.random.randn(7, 7).astype(numpy.float32)
        oinf2 = OnnxInference(onx2)
        y2 = oinf2.run({'X': x})['Y']
        oinf3 = OnnxInference(inlined)
        y3 = oinf3.run({'X': x})['Y']
        self.assertEqualArray(y2, y3)

    def test_onnx_inline_function_function(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        fft2d = os.path.join(data, "fft2d.onnx")
        onx = load(fft2d)
        fct = onnx_model_to_function(onx, name="fft2d")
        op = OnnxOperatorFunction(fct, 'X', output_names=['Y'])
        onx2 = op.to_onnx(numpy.float32, numpy.float32)

        fct = onnx_model_to_function(onx2, name="fft2d")
        inlined, m = onnx_inline_function(fct, list(onx2.functions))
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].op_type, "fft2d")
        self.assertEqual(len(inlined.node), 35)


if __name__ == "__main__":
    # TestOptimOnnxManipulations().test_onnx_inline_function()
    unittest.main()
