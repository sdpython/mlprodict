"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.verify_code import verify_code, ImperfectPythonCode

source = '''
import numpy  # pylint: disable=W0611
# Import specific to this model.
from sklearn.ensemble import VotingClassifier

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkClassifier
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611


class VotingClassifier_b_cl_logreg_noflatten_votingsoft_flatten_transfda804c_9_benchClassifier(_CommonAsvSklBenchmarkClassifier):
    """
    :epkg:`asv` test for a classifier,
    Full template can be found in
    `common_asv_skl.py <https://github.com/sdpython/mlprodict/
    blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_.
    """

    params = [
        ['skl', 'pyrt', 'ort'],
        (1, 100, 10000),
        (4, 20),
    ]
    param_names = ['rt', 'N', 'nf']
    target_opset = 9

    def setup_cache(self):
        super().setup_cache()

    def _create_model(self):
        return VotingClassifier(
            estimators=[('lr1', LogisticRegression(C=1.0, class_weight=None,
            dual=False, fit_intercept=True,
            intercept_scaling=1, l1_ratio=None, max_iter=100,
            multi_class='auto', n_jobs=None, penalty='l2',
            random_state=None, solver='liblinear', tol=0.0001, verbose=0,
            warm_start=False)), ('lr2', LogisticRegression(C=1.0,
            class_weight=None, dual=False, fit_intercept=False,
            intercept_scaling=1, l1_ratio=None, max_iter=100,
            multi_class='auto', n_jobs=None, penalty='l2',
            random_state=None, solver='liblinear', tol=0.0001, verbose=0,
            warm_start=False))], flatten_transform=False, voting='soft'
        )
'''

source2 = '''
def fct(a, b):
    return a * b
'''

source3 = '''
def fct(a, b):
    return a {} b
'''

source4 = '''
def fct(a, b):
    return [0, 1, 2, 3][a: b]
'''

source5 = '''
def fct(a, b):
    return lambda x: x * 2
'''

source6 = '''
def fct(a, b):
    return [x for x in [1, 2]]
'''

source7 = '''
def fct(a, b):
    return lambda x: {} x
'''

source8 = '''
def fct(a):
    return ([0, 1, 2, 3])[a]
'''

source9 = '''
def fct(a):
    return sum(el for el in a)
'''


class TestVerifyCode(ExtTestCase):

    def test_verify_code(self):
        self.assertRaise(lambda: verify_code(source), ImperfectPythonCode)

    def test_verify_code2(self):
        _, res = verify_code(source2)
        self.assertIn('CodeNodeVisitor', str(res))
        tree = res.print_tree()
        self.assertIn('BinOp:', tree)
        self.assertIn('\n', tree)
        rows = res.Rows
        node = rows[0]['node']
        text = res.print_node(node)
        self.assertIn('body=', text)

    def test_verify_code_ops(self):
        for op in ['**', 'and', '*', '/', '-', '+', 'or', '&', '|']:
            with self.subTest(op=op):
                _, res = verify_code(source3.format(op))
                self.assertIn('CodeNodeVisitor', str(res))
                tree = res.print_tree()
                if 'BinOp' not in tree and 'BoolOp' not in tree:
                    raise AssertionError(
                        f"Unable to find {op!r} in\n{str(tree)!r}")
                self.assertIn('\n', tree)
                rows = res.Rows
                node = rows[0]['node']
                text = res.print_node(node)
                self.assertIn('body=', text)

    def test_verify_code_cmp(self):
        for op in ['<', '>', '==', '!=', '>=', '<=', 'is', 'is not']:
            with self.subTest(op=op):
                _, res = verify_code(source3.format(op))
                self.assertIn('CodeNodeVisitor', str(res))
                tree = res.print_tree()
                self.assertIn('Compare', tree)
                self.assertIn('\n', tree)
                rows = res.Rows
                node = rows[0]['node']
                text = res.print_node(node)
                self.assertIn('body=', text)

    def test_verify_code_not(self):
        for op in ['not', '-', '+']:
            with self.subTest(op=op):
                _, res = verify_code(source7.format(op))
                self.assertIn('CodeNodeVisitor', str(res))
                tree = res.print_tree()
                self.assertIn('UnaryOp', tree)
                self.assertIn('\n', tree)
                rows = res.Rows
                node = rows[0]['node']
                text = res.print_node(node)
                self.assertIn('body=', text)

    def test_verify_code_slice(self):
        _, res = verify_code(source4)
        self.assertIn('CodeNodeVisitor', str(res))
        tree = res.print_tree()
        self.assertIn('Slice', tree)
        self.assertIn('\n', tree)
        rows = res.Rows
        node = rows[0]['node']
        text = res.print_node(node)
        self.assertIn('body=', text)

    def test_verify_code_index(self):
        _, res = verify_code(source8)
        self.assertIn('CodeNodeVisitor', str(res))
        tree = res.print_tree()
        self.assertIn('Subscript', tree)
        self.assertIn('\n', tree)
        rows = res.Rows
        node = rows[0]['node']
        text = res.print_node(node)
        self.assertIn('body=', text)

    def test_verify_code_ops_in(self):
        for op in ['in', 'not in']:
            with self.subTest(op=op):
                _, res = verify_code(source3.format(op))
                self.assertIn('CodeNodeVisitor', str(res))
                tree = res.print_tree()
                self.assertIn('Compare', tree)
                self.assertIn('\n', tree)
                rows = res.Rows
                node = rows[0]['node']
                text = res.print_node(node)
                self.assertIn('body=', text)

    def test_verify_code_lambda(self):
        _, res = verify_code(source5)
        self.assertIn('CodeNodeVisitor', str(res))
        tree = res.print_tree()
        self.assertIn('Lambda', tree)
        self.assertIn('\n', tree)
        rows = res.Rows
        node = rows[0]['node']
        text = res.print_node(node)
        self.assertIn('body=', text)

    def test_verify_code_gen(self):
        _, res = verify_code(source6, exc=False)
        self.assertIn('CodeNodeVisitor', str(res))
        tree = res.print_tree()
        self.assertIn('comprehension', tree)
        self.assertIn('\n', tree)
        rows = res.Rows
        node = rows[0]['node']
        text = res.print_node(node)
        self.assertIn('body=', text)

    def test_verify_code_gen2(self):
        _, res = verify_code(source9, exc=False)
        self.assertIn('CodeNodeVisitor', str(res))
        tree = res.print_tree()
        self.assertIn('comprehension', tree)
        self.assertIn('\n', tree)
        rows = res.Rows
        node = rows[0]['node']
        text = res.print_node(node)
        self.assertIn('body=', text)


if __name__ == "__main__":
    TestVerifyCode().test_verify_code_gen()
    unittest.main()
