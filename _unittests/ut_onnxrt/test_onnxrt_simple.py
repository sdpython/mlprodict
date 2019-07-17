"""
@brief      test log(time=2s)
"""
import os
from io import BytesIO
import pickle
import unittest
import warnings
from logging import getLogger
import numpy
import pandas
from onnx.onnx_cpp2py_export.checker import ValidationError  # pylint: disable=E0401,E0611
from onnx.helper import make_tensor
from onnx import TensorProto
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxLinearRegressor, OnnxLinearClassifier,
    OnnxConstantOfShape, OnnxShape, OnnxIdentity
)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx, __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtSimple(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_idi(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})

        oinf = OnnxInference(model_def)
        res = str(oinf)
        self.assertIn('op_type: "Add"', res)

        sb = model_def.SerializeToString()
        oinf = OnnxInference(sb)
        res = str(oinf)
        self.assertIn('op_type: "Add"', res)

        sb = BytesIO(model_def.SerializeToString())
        oinf = OnnxInference(sb)
        res = str(oinf)
        self.assertIn('op_type: "Add"', res)

        temp = get_temp_folder(__file__, "temp_onnxrt_idi")
        name = os.path.join(temp, "m.onnx")
        with open(name, "wb") as f:
            f.write(model_def.SerializeToString())

        oinf = OnnxInference(name)
        res = str(oinf)
        self.assertIn('op_type: "Add"', res)

    def test_onnxt_pickle_check(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        shape = oinf.shape_inference()
        self.assertNotEmpty(shape)
        try:
            oinf.check_model()
        except ValidationError as e:
            warnings.warn("Why? " + str(e))  # pylint: disable=E1101

        pkl = pickle.dumps(oinf)
        obj = pickle.loads(pkl)
        self.assertEqual(str(oinf), str(obj))

    def test_onnxt_dot(self):
        idi = numpy.identity(2)
        idi2 = numpy.identity(2) * 2
        onx = OnnxAdd(OnnxAdd('X', idi), idi2, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        dot = oinf.to_dot()
        self.assertIn('Add [', dot)
        self.assertIn('Add1 [', dot)
        self.assertIn('Add\\n(Add)', dot)
        self.assertIn('Add\\n(Add1)', dot)
        self.assertIn('X -> Add;', dot)
        self.assertIn('Addcst1 -> Add1;', dot)
        self.assertIn('Addcst -> Add;', dot)
        self.assertIn('Add1 -> Y;', dot)

    def test_onnxt_lreg(self):
        pars = dict(coefficients=numpy.array([1., 2.]), intercepts=numpy.array([1.]),
                    post_transform='NONE')
        onx = OnnxLinearRegressor('X', output_names=['Y'], **pars)
        model_def = onx.to_onnx({'X': pars['coefficients'].astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([1]))])
        oinf = OnnxInference(model_def)
        dot = oinf.to_dot()
        self.assertIn('coefficients=[1. 2.]', dot)
        self.assertIn('LinearRegressor', dot)

    def test_onnxt_lrc(self):
        pars = dict(coefficients=numpy.array([1., 2.]), intercepts=numpy.array([1.]),
                    classlabels_ints=[0, 1], post_transform='NONE')
        onx = OnnxLinearClassifier('X', output_names=['Y'], **pars)
        model_def = onx.to_onnx({'X': pars['coefficients'].astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([1]))])
        oinf = OnnxInference(model_def)
        dot = oinf.to_dot()
        self.assertIn('coefficients=[1. 2.]', dot)
        self.assertIn('LinearClassifier', dot)

    def test_onnxt_lrc_iris(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression(solver="liblinear")
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        dot = oinf.to_dot()
        self.assertIn('ZipMap', dot)
        self.assertIn('LinearClassifier', dot)

    def test_onnxt_lrc_iris_json(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression(solver="liblinear")
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        js = oinf.to_json()
        self.assertIn('"producer_name": "skl2onnx",', js)
        self.assertIn('"name": "output_label",', js)
        self.assertIn('"name": "output_probability",', js)
        self.assertIn('"name": "LinearClassifier",', js)
        self.assertIn('"coefficients": {', js)
        self.assertIn('"name": "Normalizer",', js)
        self.assertIn('"name": "Cast",', js)
        self.assertIn('"name": "ZipMap",', js)

    def test_onnxt_json(self):
        idi = numpy.identity(2)
        idi2 = numpy.identity(2) * 2
        onx = OnnxAdd(OnnxAdd('X', idi), idi2, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        js = oinf.to_json()
        self.assertIn('"initializers": {', js)

    def test_onnxt_graph(self):
        idi = numpy.identity(2)
        idi2 = numpy.identity(2) * 2
        onx = OnnxAdd(OnnxAdd('X', idi), idi2, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        js = oinf.to_sequence()
        self.assertIn('inits', js)
        self.assertIn('inputs', js)
        self.assertIn('outputs', js)
        self.assertIn('intermediate', js)
        self.assertIn('nodes', js)
        self.assertIn('sequence', js)
        self.assertEqual(len(js['sequence']), 2)
        self.assertEqual(len(js['intermediate']), 2)

    def test_onnxt_run(self):
        idi = numpy.identity(2)
        idi2 = numpy.identity(2) * 2
        onx = OnnxAdd(OnnxAdd('X', idi), idi2, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        X = numpy.array([[1, 1], [3, 3]])
        y = oinf.run({'X': X})
        exp = numpy.array([[4, 1], [3, 6]], dtype=numpy.float32)
        self.assertEqual(list(y), ['Y'])
        self.assertEqualArray(y['Y'], exp)

    def test_onnxt_lrreg_iris_run(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LinearRegression()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        y = oinf.run({'X': X_test})
        exp = clr.predict(X_test)
        self.assertEqual(list(sorted(y)), ['variable'])
        self.assertEqualArray(exp, y['variable'].ravel(), decimal=6)

    def test_onnxt_lrc_iris_run(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression(solver="liblinear")
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['output_label'])

        exp = clr.predict_proba(X_test)
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_onnxt_knn_iris_dot(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = KNeighborsClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def, skip_run=True)
        dot = oinf.to_dot()
        self.assertNotIn("class_labels_0 -> ;", dot)

    def test_getitem(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = KNeighborsClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def, skip_run=True)

        topk = oinf['ArrayFeatureExtractor']
        self.assertIn('ArrayFeatureExtractor', str(topk))
        zm = oinf['ZipMap']
        self.assertIn('ZipMap', str(zm))
        par = oinf['ZipMap', 'classlabels_int64s']
        self.assertIn('classlabels_int64s', str(par))

    def test_constant_of_shape(self):
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        tensor_value = make_tensor(
            "value", TensorProto.FLOAT, (1,), [-5])  # pylint: disable=E1101
        cop2 = OnnxConstantOfShape(OnnxShape('input'), value=tensor_value,
                                   output_names=['mat'])
        model_def = cop2.to_onnx({'input': x},
                                 outputs=[('mat', FloatTensorType())])
        oinf = OnnxInference(model_def, skip_run=True)
        dot = oinf.to_dot()
        self.assertIn('ConstantOfShape', dot)

    def test_onnxt_pdist_dot(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist  # pylint: disable=E0401,E0611
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('input', 'input')
        cdist = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': x}, outputs=[('cdist', FloatTensorType())])

        oinf = OnnxInference(model_def, skip_run=True)
        dot = oinf.to_dot(recursive=True)
        self.assertIn("B_next_out", dot)
        self.assertIn("cluster", dot)

    def test_onnxt_lrc_iris_run_node_time(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression(solver="liblinear")
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        _, mt = oinf.run({'X': X_test}, node_time=True)
        self.assertIsInstance(mt, list)
        self.assertGreater(len(mt), 1)
        self.assertIsInstance(mt[0], dict)

        rows = []

        def myprint(*args):
            rows.append(' '.join(map(str, args)))

        _, mt = oinf.run({'X': X_test}, node_time=True,
                         verbose=1, fLOG=myprint)
        self.assertIsInstance(mt, list)
        self.assertGreater(len(mt), 1)
        self.assertIsInstance(mt[0], dict)


if __name__ == "__main__":
    unittest.main()
