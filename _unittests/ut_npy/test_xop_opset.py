# pylint: disable=E0611
"""
@brief      test log(time=15s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop, OnnxOperatorFunction
from mlprodict.npy.xop_variable import Variable


class TestXOpsOpset(ExtTestCase):

    def test_onnx_function_init(self):
        opset = 17
        OnnxAbs, OnnxAdd, OnnxDiv = loadop("Abs", "Add", "Div")
        ov = OnnxAbs[opset]('X')
        ad = OnnxAdd[opset]('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')
        op = OnnxDiv[opset](OnnxOperatorFunction(proto, 'X'),
                            numpy.array([2], dtype=numpy.float32),
                            output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

    def test_onnx_function_wrong(self):
        OnnxCos = loadop("Cos")
        self.assertRaise(lambda: OnnxCos[1]('X'), ValueError)
        self.assertRaise(lambda: OnnxCos['R']('X'), ValueError)

    def test_opset_scan_body(self):
        from mlprodict.npy.xop_opset import OnnxReduceSumSquareApi18
        (OnnxSub, OnnxIdentity, OnnxScan, OnnxAdd) = loadop(
            'Sub', 'Identity', 'Scan', 'Add')

        # Building of the subgraph.
        opv = 18
        diff = OnnxSub('next_in', 'next', op_version=opv)
        id_next = OnnxIdentity('next_in', output_names=['next_out'],
                               op_version=opv)
        flat = OnnxReduceSumSquareApi18(
            diff, axes=[1], output_names=['scan_out'], keepdims=0,
            op_version=opv)
        scan_body = id_next.to_onnx(
            [Variable('next_in', numpy.float32, (None, None)),
             Variable('next', numpy.float32, (None, ))],
            outputs=[Variable('next_out', numpy.float32, (None, None)),
                     Variable('scan_out', numpy.float32, (None, ))],
            other_outputs=[flat], target_opset=opv)
        opsets1 = {d.domain: d.version for d in scan_body.opset_import}
        output_names = [o.name for o in scan_body.graph.output]

        cop = OnnxAdd('input', 'input', op_version=opv)

        # Subgraph as a graph attribute.
        node = OnnxScan(cop, cop, output_names=['S1', 'S2'],
                        num_scan_inputs=1,
                        body=(scan_body.graph, [id_next, flat]),
                        op_version=opv)

        cop2 = OnnxIdentity(node[1], output_names=['cdist'], op_version=opv)

        model_def = cop2.to_onnx(numpy.float32, numpy.float32,
                                 target_opset=opv)
        opsets2 = {d.domain: d.version for d in model_def.opset_import}
        self.assertGreater(opsets2[''], opsets1[''])

        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        sess = OnnxInference(model_def)
        res = sess.run({'input': x})
        self.assertEqual(res['cdist'].shape, (3, 3)) 


if __name__ == "__main__":
    unittest.main(verbosity=2)
