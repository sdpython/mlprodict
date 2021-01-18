"""
@brief      test tree node (time=30s)
"""
import os
import unittest
import pickle
from pyquickhelper.loghelper import fLOG
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase, get_temp_folder, skipif_circleci
from mlprodict.__main__ import main


class TestCliValidateDump(ExtTestCase):

    @skipif_circleci('too long')
    def test_cli_validate_model_dump(self):
        fLOG(OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_validate_model_dump")
        out1 = os.path.join(temp, "raw.csv")
        out2 = os.path.join(temp, "sum.csv")
        graph = os.path.join(temp, 'benchmark.png')
        st = BufferedPrint()
        models = ','.join([
            "LinearRegression",
            # "LogisticRegression",
            "DecisionTreeRegressor",
            # "DecisionTreeClassifier",
        ])
        # ~ models = ','.join([
        #~ 'KMeans',
        #~ 'LGBMClassifier',
        #~ 'LGBMRegressor',
        #~ 'LinearSVC',
        #~ 'LogisticRegression',
        #~ 'MLPClassifier',
        #~ 'MLPRegressor',
        #~ 'RandomForestClassifier',
        #~ 'Perceptron',
        #~ 'RandomForestClassifier',
        #~ 'Ridge',
        #~ 'SGDRegressor',
        #~ 'RandomForestRegressor',
        # ~ ])
        args = ["validate_runtime", "--out_raw", out1,
                "--out_summary", out2, "--models",
                models, '-r', "python,onnxruntime1",
                '-o', '10', '-op', '10', '-v', '1', '-b', '1',
                '-dum', '1', '-du', temp, '-n', '20,50',
                '--out_graph', graph, '--dtype', '32']
        cmd = "python -m mlprodict " + " ".join(args)
        fLOG(cmd)
        main(args=args, fLOG=fLOG if __name__ == "__main__" else st.fprint)
        names = os.listdir(temp)
        names = [_ for _ in names if "dump-i-" in _]
        self.assertNotEmpty(names)
        for i, name in enumerate(names):
            fLOG("{}/{}: {}".format(i + 1, len(names), name))
            fullname = os.path.join(temp, name)
            with open(fullname, 'rb') as f:
                pkl = pickle.load(f)
            root = os.path.splitext(fullname)[0]
            with open(root + '.onnx', 'wb') as f:
                f.write(pkl['onnx_bytes'])
            with open(root + '.data.pkl', 'wb') as f:
                pickle.dump(pkl['Xort_test'], f)
            with open(root + '.ypred.pkl', 'wb') as f:
                pickle.dump(pkl['ypred'], f)
            with open(root + '.skl.pkl', 'wb') as f:
                pickle.dump(pkl['skl_model'], f)


if __name__ == "__main__":
    unittest.main()
