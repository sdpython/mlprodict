"""
@brief      test log(time=6s)
"""
import unittest
import io
from contextlib import redirect_stdout
import itertools
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.einsum_impl_ext import (
    numpy_diagonal, numpy_extended_dot, numpy_extended_dot_python)
from mlprodict.testing.einsum_impl import (
    analyse_einsum_equation, decompose_einsum_equation, EinsumSubOp,
    apply_einsum_sequence)


class TestEinsum(ExtTestCase):

    def test_numpy_diagonal(self):
        mat = numpy.arange(8).reshape((2, 2, 2))
        diag = numpy_diagonal(mat, 1, [1, 2])
        self.assertEqualArray(diag, numpy.array([[0, 3], [4, 7]]))
        diag = numpy_diagonal(mat, 2, [1, 2])
        self.assertEqualArray(diag, numpy.array([[0, 3], [4, 7]]))

        diag = numpy_diagonal(mat, 0, [0, 1])
        self.assertEqualArray(diag, numpy.array([[0, 1], [6, 7]]))
        diag = numpy_diagonal(mat, 1, [0, 1])
        self.assertEqualArray(diag, numpy.array([[0, 1], [6, 7]]))

        diag = numpy_diagonal(mat, 0, [0, 2])
        self.assertEqualArray(diag, numpy.array([[0, 2], [5, 7]]))
        diag = numpy_diagonal(mat, 2, [0, 2])
        self.assertEqualArray(diag, numpy.array([[0, 2], [5, 7]]).T)

    def test_numpy_extended_dot_2_a(self):
        m1 = numpy.arange(4).reshape((2, 2)).astype(numpy.float32) + 10
        m2 = m1 + 90

        self.assertRaise(lambda: numpy_extended_dot(m1, m2.T, [0], [1], [2]),
                         ValueError)
        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[1], left=[0], right=[2])
        exp = m1 @ m2
        self.assertEqualArray(exp, numpy.squeeze(dot))
        dot2 = numpy_extended_dot_python(
            dm1, dm2, axes=[1], left=[0], right=[2])
        self.assertEqualArray(exp, numpy.squeeze(dot2))

        dm1 = m1.reshape((2, 1, 2))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0], right=[1])
        exp = m1 @ m2.T
        self.assertEqualArray(exp, numpy.squeeze(dot))
        dot2 = numpy_extended_dot_python(
            dm1, dm2, axes=[2], left=[0], right=[1])
        self.assertEqualArray(exp, numpy.squeeze(dot2))

    def test_numpy_extended_dot_2_b(self):
        m1 = numpy.arange(4).reshape((2, 2)).astype(numpy.float32) + 10
        m2 = m1 + 90
        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0], right=[1, 2])
        dot2 = numpy_extended_dot_python(
            dm1, dm2, axes=[2], left=[0], right=[1, 2])
        self.assertEqualArray(dot, numpy.squeeze(dot2))

    def test_numpy_extended_dot_2_b2(self):
        m1 = numpy.arange(4).reshape((2, 2)).astype(numpy.float32) + 10
        m2 = m1 + 90
        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0, 1], right=[2])
        dot2 = numpy_extended_dot_python(
            dm1, dm2, axes=[2], left=[0, 1], right=[2])
        self.assertEqualArray(dot, numpy.squeeze(dot2))

    def test_numpy_extended_dot_3(self):
        m1 = numpy.arange(8).reshape((2, 2, 2)) + 10
        m2 = m1 + 90

        dot = numpy_extended_dot(m1, m2, [1], [0], [2])
        dot2 = numpy_extended_dot_python(m1, m2, [1], [0], [2])
        self.assertEqualArray(dot, dot2)

        dot = numpy_extended_dot(m1, m2, [1], [2], [0])
        dot2 = numpy_extended_dot_python(m1, m2, [1], [2], [0])
        self.assertEqualArray(dot, dot2)

    def test_numpy_extended_dot_3b(self):
        m1 = numpy.arange(8).reshape((2, 2, 2)) + 10
        m2 = m1 + 90

        dot = numpy_extended_dot(m1, m2, [1], [2], [0, 1])
        dot2 = numpy_extended_dot_python(m1, m2, [1], [2], [0, 1])
        self.assertEqualArray(dot, dot2)

    def test_analyse_einsum_equation(self):
        self.assertRaise(lambda: analyse_einsum_equation("abc"),
                         NotImplementedError)
        self.assertRaise(lambda: analyse_einsum_equation("abc0,ch->ah"),
                         ValueError)
        self.assertRaise(lambda: analyse_einsum_equation("abc,ch->a0"),
                         ValueError)
        res = analyse_einsum_equation("abc,ch->ah")
        self.assertEqual(len(res), 4)
        letters, mat, lengths, duplicates = res
        self.assertEqual(letters, "abch")
        self.assertEqualArray(lengths, numpy.array([3, 2, 2]))
        self.assertEqualArray(
            mat, numpy.array([[0, 1, 2, -1],
                              [-1, -1, 0, 1],
                              [0, -1, -1, 1]]))
        self.assertEqual(duplicates, [None, None, None])

    def test_analyse_einsum_equation_duplicates(self):
        res = analyse_einsum_equation("aac,ca->aa")
        self.assertEqual(len(res), 4)
        letters, mat, lengths, duplicates = res
        self.assertEqual(letters, "ac")
        self.assertEqualArray(lengths, numpy.array([3, 2, 2]))
        self.assertEqual(duplicates, [{'a': [0, 1], 'c': [2]},
                                      None,
                                      {'a': [0, 1]}])
        self.assertEqualArray(
            mat, numpy.array([[1, 2],
                              [1, 0],
                              [1, -1]]))

    def test_decompose_einsum_equation_exc(self):
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2, 2), (2, 2),
                                              strategy="donotexist"),
            ValueError)
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah"), ValueError)
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2, 2), (2, 2),
                                              "donotexist"),
            TypeError)
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2, 2)),
            ValueError)
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2), (2, 2)),
            ValueError)

    def test_decompose_einsum_equation(self):
        m1 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2))
        m2 = numpy.arange(0, 4).astype(numpy.float32).reshape((2, 2))
        exp = numpy.einsum("bac,ch->ah", m1, m2)

        def fct():
            print("########################## DECOMPOSE")
            seq = decompose_einsum_equation(
                "bac,ch->ah", (2, 2, 2), (2, 2), verbose=True)
            print("########################## APPLY")
            dot = seq.to_dot()
            print(dot)
            red = dot.split('red')
            self.assertEqual(len(red), 4)
            res = apply_einsum_sequence(seq, m1, m2, verbose=True)
            print("########################## END")
            return res

        f = io.StringIO()
        try:
            with redirect_stdout(f):
                res = fct()
        except Exception as e:
            raise AssertionError("Issue. Logs =\n%s" % f.getvalue()) from e

        out = f.getvalue()
        self.assertIn("numpy_extended_dot", out)
        self.assertEqualArray(exp, res)

    def test_decompose_einsum_equation_py(self):
        m1 = numpy.arange(0, 24).astype(numpy.float32).reshape((2, 3, 4))
        m2 = numpy.arange(0, 20).astype(numpy.float32).reshape((4, 5))
        verbose = False
        for strat in ['numpy', 'simple']:
            with self.subTest(strategy=strat):
                seq = decompose_einsum_equation(
                    "bac,ch->ah", (2, 3, 4), (4, 5), strategy=strat,
                    verbose=verbose)
                res1 = apply_einsum_sequence(seq, m1, m2, verbose=verbose)
                res2 = apply_einsum_sequence(
                    seq, m1, m2, matmul_impl='py', verbose=verbose)
                if strat == 'simple':
                    self.assertRaise(
                        lambda: apply_einsum_sequence(
                            seq, m1, m2, matmul_impl='py2'),  # pylint: disable=W0640
                        ValueError)
                self.assertEqualArray(res1, res2)

    def test_decompose_einsum_equation_pyf(self):
        m1 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2))
        m2 = numpy.arange(0, 4).astype(numpy.float32).reshape((2, 2))
        seq = decompose_einsum_equation(
            "bac,ch->ah", (2, 2, 2), (2, 2))
        res1 = apply_einsum_sequence(seq, m1, m2)
        res2 = apply_einsum_sequence(seq, m1, m2, matmul_impl='pyf')
        self.assertEqualArray(res1, res2)

    def test_einsum_sub_op(self):
        self.assertRaise(lambda: EinsumSubOp(2, "er", (2, 2)), ValueError)
        self.assertRaise(lambda: EinsumSubOp(2, "expand_dims"), RuntimeError)
        self.assertRaise(lambda: EinsumSubOp(
            2, "matmul", (2, 2)), RuntimeError)
        self.assertRaise(lambda: EinsumSubOp(2, "id", (2, 2)), TypeError)

    def test_case_1_iii_ii_i(self):
        verbose = False
        equation = 'ii->i'
        m1 = numpy.arange(2 * 2).reshape((2, 2)) + 10
        exp = numpy.einsum(equation, m1)
        seq = decompose_einsum_equation(
            equation, m1.shape, verbose=verbose)
        res = apply_einsum_sequence(seq, m1, verbose=verbose)
        self.assertEqualArray(exp, res)

    def test_case_1_iii_ii_i_j(self):
        verbose = False
        equation = 'iij->ij'
        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        exp = numpy.einsum(equation, m1)
        seq = decompose_einsum_equation(
            equation, m1.shape, verbose=verbose)
        dot = seq.to_dot()
        self.assertIn("i=0,1", dot)
        res = apply_einsum_sequence(seq, m1, verbose=verbose)
        self.assertEqualArray(exp, res)

    def common_test_case_2(self, equation, verbose=False, strategy='simple'):
        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = numpy.arange(4).reshape((2, 2)) + 100
        exp = numpy.einsum(equation, m1, m2)

        seq = decompose_einsum_equation(
            equation, m1.shape, m2.shape, verbose=verbose, strategy=strategy)
        res = apply_einsum_sequence(seq, m1, m2, verbose=verbose)
        self.assertEqualArray(exp, res)

    def test_case_2_A(self):
        for strat in ['numpy', 'simple']:
            with self.subTest(strategy=strat):
                self.common_test_case_2(
                    'abc,cd->abc', strategy=strat, verbose=False)

    def test_many_2(self):
        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = numpy.arange(4).reshape((2, 2)) + 100

        res = []
        for p1 in itertools.permutations(list("abc")):
            for p2 in itertools.permutations(list("cd")):
                for i in [1, 2]:
                    for j in [0, 1]:
                        sp1 = "".join(p1)
                        sp2 = "".join(p2)
                        if len(set([sp1[0], sp1[i], sp2[j]])) != 3:
                            continue
                        equation = "%s,%s->%s%s%s" % (
                            sp1, sp2, sp1[0], sp1[i], sp2[j])
                        try:
                            r = numpy.einsum(equation, m1, m2)
                            res.append((equation, r))
                        except ValueError:
                            # Not viable equation.
                            continue

        for i, (eq, exp) in enumerate(res):
            with self.subTest(equation=eq, index=i, total=len(res)):
                seq = decompose_einsum_equation(
                    eq, m1.shape, m2.shape)
                res = apply_einsum_sequence(seq, m1, m2)
                self.assertEqualArray(exp, res)

    def test_many_3(self):
        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = numpy.arange(4).reshape((2, 2)) + 100
        m3 = numpy.arange(8).reshape((2, 2, 2)) + 1000

        res = []
        for p1 in itertools.permutations(list("abc")):  # pylint: disable=R1702
            for p2 in itertools.permutations(list("cd")):
                for p3 in itertools.permutations(list("def")):
                    for i in [1, 2]:
                        for j in [0, 1]:
                            sp1 = "".join(p1)
                            sp2 = "".join(p2)
                            sp3 = "".join(p3)
                            equation = "%s,%s,%s->%s%s%s" % (
                                sp1, sp2, sp3, sp1[0], sp1[i], sp3[j])
                            try:
                                r = numpy.einsum(equation, m1, m2, m3)
                                res.append((equation, r))
                            except ValueError:
                                # Not viable equation.
                                continue

        for i, (eq, exp) in enumerate(res):
            with self.subTest(equation=eq, index=i, total=len(res)):
                seq = decompose_einsum_equation(
                    eq, m1.shape, m2.shape, m3.shape)
                res = apply_einsum_sequence(seq, m1, m2, m3)
                self.assertEqualArray(exp, res)

    # Taken from https://github.com/numpy/numpy/blob/main/numpy/
    # core/tests/test_einsum.py.

    def optimize_compare(self, equation, operands=None, verbose=False):
        with self.subTest(equation=equation):
            if operands is not None:
                inputs = operands
            else:
                eqs = equation.split("->")[0].split(",")
                inputs = []
                for d, eq in enumerate(eqs):
                    i = numpy.arange(2 ** len(eq)).reshape(
                        (2,) * len(eq)).astype(numpy.float32)
                    inputs.append(
                        i + numpy.array([3 ** d], dtype=numpy.float32))

            exp = numpy.einsum(equation, *inputs)
            if verbose:
                print("###### equation", equation)
                path = numpy.einsum_path(equation, *inputs, optimize=False)
                print(path[1])
                path = numpy.einsum_path(equation, *inputs)
                print(path[1])

            shapes = [m.shape for m in inputs]

            with self.subTest(strategy='numpy'):
                seq = decompose_einsum_equation(
                    equation, *shapes, verbose=verbose, strategy='numpy')
                got = apply_einsum_sequence(seq, *inputs, verbose=verbose)
                self.assertEqualArray(exp, got, decimal=6)
            with self.subTest(strategy='simple'):
                seq = decompose_einsum_equation(
                    equation, *shapes, verbose=verbose)
                got = apply_einsum_sequence(seq, *inputs, verbose=verbose)
                self.assertEqualArray(exp, got, decimal=6)

    def test_numpy_test_hadamard_like_products(self):
        # Hadamard outer products
        self.optimize_compare('a,ab,abc->abc')
        self.optimize_compare('a,b,ab->ab')

    def test_np_test_np_test_collapse(self):
        # Inner products
        self.optimize_compare('ab,ab,cd,cd->ac')
        self.optimize_compare('ab,ab,c->c')
        self.optimize_compare('ab,ab,cd,cd->cd')
        # self.optimize_compare('ab,ab,c->')
        # self.optimize_compare('ab,ab,cd,cd->')
        # self.optimize_compare('ab,ab,cd,cd,ef,ef->')

    def test_np_test_index_transformations(self):
        # Simple index transformation cases
        self.optimize_compare('ea,fb,gc,hd,abcd->efgh')
        self.optimize_compare('ea,fb,abcd,gc,hd->efgh')
        self.optimize_compare('abcd,ea,fb,gc,hd->efgh')

    def test_np_test_expand(self):
        # Outer products
        self.optimize_compare('ab,cd,ef->abcdef')
        self.optimize_compare('ab,cd,ef->acdf')
        self.optimize_compare('ab,cd,de->abcde')
        self.optimize_compare('ab,cd,de->be')
        self.optimize_compare('ab,bcd,cd->abcd')
        self.optimize_compare('ab,bcd,cd->abd')

    def test_np_test_edge_cases1(self):
        # Difficult edge cases for optimization
        self.optimize_compare('efc,dbc,acf,fd->abe', verbose=False)
        self.optimize_compare(
            'eac->ace', operands=[numpy.arange(24).reshape((2, 3, 4))])
        self.optimize_compare('eac->ace')
        self.optimize_compare('bd,db,eac->ace')
        self.optimize_compare('ba,ac,da->bcd')

    def test_np_test_edge_cases2(self):
        # Difficult edge cases for optimization
        self.optimize_compare(
            'eac->ace', operands=[numpy.arange(24).reshape((2, 3, 4))])
        self.optimize_compare('eb,cb,fb->cef')

    def test_np_test_random_cases(self):
        # Randomly built test cases
        self.optimize_compare('aab,fa,df,ecc->bde')
        self.optimize_compare('bb,ff,be->e')
        self.optimize_compare('afd,ba,cc,dc->bf')
        self.optimize_compare('bbd,bda,fc,db->acf')
        self.optimize_compare('dba,ead,cad->bce')
        self.optimize_compare('aef,fbc,dca->bde')

    def test_np_test_combined_views_mapping(self):
        # gh-10792
        a = numpy.arange(9).reshape(1, 1, 3, 1, 3)
        b = numpy.einsum('bbcdc->d', a)
        self.assertEqualArray(b, [12])

    def test_np_test_broadcasting_dot_cases1(self):
        # Ensures broadcasting cases are not mistaken for GEMM
        a = numpy.random.rand(1, 5, 4)
        b = numpy.random.rand(4, 6)
        c = numpy.random.rand(5, 6)
        d = numpy.random.rand(10)
        self.optimize_compare('ijk,kl,jl,i->i', operands=[a, b, c, d])

        e = numpy.random.rand(1, 1, 5, 4)
        f = numpy.random.rand(7, 7)
        self.optimize_compare('abjk,kl,jl,ab->ab', operands=[e, b, c, f])

    def test_np_test_broadcasting_dot_cases2(self):
        # Edge case found in gh-11308
        f = numpy.arange(7 * 55).reshape(7, 11, 5)
        g = numpy.arange(30).reshape(2, 3, 5)
        self.optimize_compare('obk,ijk->ioj', operands=[f, g])

    def np_test_complex(self):
        # Long test cases
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('cd,bdhe,aidb,hgca,gc,hgibcd,hgac')
        self.optimize_compare('abhe,hidj,jgba,hiab,gab')
        self.optimize_compare('bde,cdh,agdb,hica,ibd,hgicd,hiac')
        self.optimize_compare('chd,bde,agbc,hiad,hgc,hgi,hiad')
        self.optimize_compare('chd,bde,agbc,hiad,bdi,cgh,agdb')
        self.optimize_compare('bdhe,acad,hiab,agac,hibd')

    def np_test_inner_product(self):
        # Inner products
        self.optimize_compare('ab,ab')
        self.optimize_compare('ab,ba')
        self.optimize_compare('abc,abc')
        self.optimize_compare('abc,bac')
        self.optimize_compare('abc,cba')

    def test_np_test_random_cases_difficult(self):
        self.optimize_compare('db,bc,cfc->d', verbose=False)
        self.optimize_compare('cac,c,h->h')
        self.optimize_compare('cfc,c,h->h')
        self.optimize_compare('cfc,c,d->d')
        self.optimize_compare('c,cfc,d->d')
        self.optimize_compare('d,c,cfc->d')
        self.optimize_compare('d,bc,cfc->d')
        self.optimize_compare('adb,bc,cfc->d')
        self.optimize_compare('adb,bc,fa,cfc->d')
        self.optimize_compare('ecb,fef,bad,ed->ac')
        self.optimize_compare('fdf,cdd,ccd,afe->ae')
        self.optimize_compare('adb,cfc->d')

    def test_np_test_edge_cases_duplicate_indices(self):
        # Difficult edge cases for optimization
        # self.optimize_compare('bca,cdb,dbf,afc->')
        self.optimize_compare('dd,fb,be,cdb->cef')
        self.optimize_compare('dcc,fce,ea,dbf->ab')
        # self.optimize_compare('abcd,ad')
        self.optimize_compare('ed,fcd,ff,bcf->be')
        self.optimize_compare('baa,dcf,af,cde->be')
        self.optimize_compare('fff,fae,bef,def->abd')


if __name__ == "__main__":
    # TestEinsum().test_np_test_broadcasting_dot_cases1()
    unittest.main()
