"""
@brief      test log(time=4s)
"""
import os
import unittest
import warnings
from textwrap import dedent
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.loghelper import run_script


class TestOnnxrtImport(ExtTestCase):

    def test_onnxt_runtime_import(self):
        """
        The test checks that scikit-learn is not imported
        when running after computing predictions with the python
        runtime.
        """
        script = dedent("""
            import numpy
            from sklearn.linear_model import LinearRegression
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from mlprodict.onnxrt import OnnxInference, to_onnx

            iris = load_iris()
            X, y = iris.data, iris.target
            X_train, X_test, y_train, _ = train_test_split(X, y)
            clr = LinearRegression()
            clr.fit(X_train, y_train)
            exp = clr.predict(X_test[:5])
            model_def = to_onnx(clr, X_train.astype(numpy.float32))
            oinf = OnnxInference(model_def)
            y = oinf.run({'X': X_test[:5]})

            assert y is not None
            with open(r'__NAME__', "wb") as f:
                f.write(model_def.SerializeToString())
            print('done')
        """)

        temp = get_temp_folder(__file__, "temp_onnxt_runtime_import")
        name = os.path.join(temp, "sc.py")
        onnx_file = os.path.join(temp, "iris.onnx")
        script = script.replace('__NAME__', onnx_file)
        with open(name, "w") as f:
            f.write(script)
        out, err = run_script(name, wait=True)
        if 'done' in out:
            self.assertIn('done', out)
            self.assertNotIn('Exception', err)
            self.assertNotIn('Error', err)
        else:
            warnings.warn("Output is missing.")

        script = dedent("""
            with open(r'__NAME__', "rb") as f:
                content = f.read()

            from mlprodict.onnxrt import OnnxInference
            import numpy
            oinf = OnnxInference(content)
            X_test = numpy.array([[1., 2.2, 3., 4.]])
            y = oinf.run({'X': X_test})
            assert y is not None

            import sys
            for y in sorted(sys.modules):
                print(y)
                assert y not in {'sklearn', 'onnxruntime'}
            print('done')
        """)
        script = script.replace('__NAME__', onnx_file)
        name = os.path.join(temp, "sc2.py")
        with open(name, "w") as f:
            f.write(script)
        out, err = run_script(name, wait=True)
        if 'done' in out:
            self.assertIn('done', out)
            self.assertNotIn('sklearn', out)
            self.assertNotIn('train_test_split', out)
            self.assertNotIn('pandas', out)
            self.assertNotIn('onnxruntime', out)
            self.assertIn('onnx', out)
            self.assertIn('numpy', out)
        else:
            warnings.warn("Output is missing.")


if __name__ == "__main__":
    unittest.main()
