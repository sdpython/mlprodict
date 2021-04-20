"""
@brief      test log(time=6s)
"""
import unittest
import io
from contextlib import redirect_stdout
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.einsum_impl import (
    analyse_einsum_equation, decompose_einsum_equation, EinsumSubOp,
    apply_sequence)


class TestEinsum(ExtTestCase):

    def test_analyse_einsum_equation(self):
        self.assertRaise(lambda: analyse_einsum_equation("abc"),
                         NotImplementedError)
        self.assertRaise(lambda: analyse_einsum_equation("abc0,ch->ah"),
                         ValueError)
        self.assertRaise(lambda: analyse_einsum_equation("abc,ch->a0"),
                         ValueError)
        res = analyse_einsum_equation("abc,ch->ah")
        self.assertEqual(len(res), 3)
        letters, mat, lengths = res
        self.assertEqual(letters, "abch")
        self.assertEqualArray(lengths, numpy.array([3, 2, 2]))
        self.assertEqualArray(
            mat, numpy.array([[0, 1, 2, -1],
                              [-1, -1, 0, 1],
                              [0, -1, -1, 1]]))

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
        self.assertRaise(
            lambda: decompose_einsum_equation("aac,ch->ah", (2, 2), (2, 2)),
            NotImplementedError)

    def test_decompose_einsum_equation(self):
        m1 = numpy.arange(0, 8).astype(numpy.float32).reshape((2, 2, 2))
        m2 = numpy.arange(0, 4).astype(numpy.float32).reshape((2, 2))
        exp = numpy.einsum("bac,ch->ah", m1, m2)

        f = io.StringIO()
        with redirect_stdout(f):
            seq = decompose_einsum_equation(
                "bac,ch->ah", (2, 2, 2), (2, 2), verbose=True)
            res = apply_sequence(seq, m1, m2, verbose=True)
            import pprint
            pprint.pprint(seq)

        out = f.getvalue()
        print(out)
        self.assertEqual(exp, res)

    def test_einsum_sub_op(self):
        self.assertRaise(lambda: EinsumSubOp("er", (2, 2)), ValueError)
        self.assertRaise(lambda: EinsumSubOp("reshape"), RuntimeError)
        self.assertRaise(lambda: EinsumSubOp("gemm", (2, 2)), RuntimeError)
        self.assertRaise(lambda: EinsumSubOp("id", (2, 2)), TypeError)

    # Taken from https://github.com/numpy/numpy/blob/main/numpy/
    # core/tests/test_einsum.py.

    def _test_hadamard_like_products(self):
        # Hadamard outer products
        self.optimize_compare('a,ab,abc->abc')
        self.optimize_compare('a,b,ab->ab')

    def _test_index_transformations(self):
        # Simple index transformation cases
        self.optimize_compare('ea,fb,gc,hd,abcd->efgh')
        self.optimize_compare('ea,fb,abcd,gc,hd->efgh')
        self.optimize_compare('abcd,ea,fb,gc,hd->efgh')

    def _test_complex(self):
        # Long test cases
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('cd,bdhe,aidb,hgca,gc,hgibcd,hgac')
        self.optimize_compare('abhe,hidj,jgba,hiab,gab')
        self.optimize_compare('bde,cdh,agdb,hica,ibd,hgicd,hiac')
        self.optimize_compare('chd,bde,agbc,hiad,hgc,hgi,hiad')
        self.optimize_compare('chd,bde,agbc,hiad,bdi,cgh,agdb')
        self.optimize_compare('bdhe,acad,hiab,agac,hibd')

    def _test_collapse(self):
        # Inner products
        self.optimize_compare('ab,ab,c->')
        self.optimize_compare('ab,ab,c->c')
        self.optimize_compare('ab,ab,cd,cd->')
        self.optimize_compare('ab,ab,cd,cd->ac')
        self.optimize_compare('ab,ab,cd,cd->cd')
        self.optimize_compare('ab,ab,cd,cd,ef,ef->')

    def _test_expand(self):
        # Outer products
        self.optimize_compare('ab,cd,ef->abcdef')
        self.optimize_compare('ab,cd,ef->acdf')
        self.optimize_compare('ab,cd,de->abcde')
        self.optimize_compare('ab,cd,de->be')
        self.optimize_compare('ab,bcd,cd->abcd')
        self.optimize_compare('ab,bcd,cd->abd')

    def _test_edge_cases(self):
        # Difficult edge cases for optimization
        self.optimize_compare('eb,cb,fb->cef')
        self.optimize_compare('dd,fb,be,cdb->cef')
        self.optimize_compare('bca,cdb,dbf,afc->')
        self.optimize_compare('dcc,fce,ea,dbf->ab')
        self.optimize_compare('fdf,cdd,ccd,afe->ae')
        self.optimize_compare('abcd,ad')
        self.optimize_compare('ed,fcd,ff,bcf->be')
        self.optimize_compare('baa,dcf,af,cde->be')
        self.optimize_compare('bd,db,eac->ace')
        self.optimize_compare('fff,fae,bef,def->abd')
        self.optimize_compare('efc,dbc,acf,fd->abe')
        self.optimize_compare('ba,ac,da->bcd')

    def _test_inner_product(self):
        # Inner products
        self.optimize_compare('ab,ab')
        self.optimize_compare('ab,ba')
        self.optimize_compare('abc,abc')
        self.optimize_compare('abc,bac')
        self.optimize_compare('abc,cba')

    def _test_random_cases(self):
        # Randomly built test cases
        self.optimize_compare('aab,fa,df,ecc->bde')
        self.optimize_compare('ecb,fef,bad,ed->ac')
        self.optimize_compare('bcf,bbb,fbf,fc->')
        self.optimize_compare('bb,ff,be->e')
        self.optimize_compare('bcb,bb,fc,fff->')
        self.optimize_compare('fbb,dfd,fc,fc->')
        self.optimize_compare('afd,ba,cc,dc->bf')
        self.optimize_compare('adb,bc,fa,cfc->d')
        self.optimize_compare('bbd,bda,fc,db->acf')
        self.optimize_compare('dba,ead,cad->bce')
        self.optimize_compare('aef,fbc,dca->bde')

    def _test_combined_views_mapping(self):
        # gh-10792
        a = numpy.arange(9).reshape(1, 1, 3, 1, 3)
        b = numpy.einsum('bbcdc->d', a)
        assert_equal(b, [12])

    def _test_broadcasting_dot_cases(self):
        # Ensures broadcasting cases are not mistaken for GEMM

        a = numpy.random.rand(1, 5, 4)
        b = numpy.random.rand(4, 6)
        c = numpy.random.rand(5, 6)
        d = numpy.random.rand(10)

        self.optimize_compare('ijk,kl,jl', operands=[a, b, c])
        self.optimize_compare('ijk,kl,jl,i->i', operands=[a, b, c, d])

        e = numpy.random.rand(1, 1, 5, 4)
        f = numpy.random.rand(7, 7)
        self.optimize_compare('abjk,kl,jl', operands=[e, b, c])
        self.optimize_compare('abjk,kl,jl,ab->ab', operands=[e, b, c, f])

        # Edge case found in gh-11308
        g = numpy.arange(64).reshape(2, 4, 8)
        self.optimize_compare('obk,ijk->ioj', operands=[g, g])


if __name__ == "__main__":
    # TestEinsum().test_decompose_einsum_equation()
    # stop
    unittest.main()
