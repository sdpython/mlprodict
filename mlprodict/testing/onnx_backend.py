"""
@file
@brief Tests with onnx backend.
"""
import os
import textwrap
from numpy.testing import assert_almost_equal
import onnx
from onnx.numpy_helper import to_array
from onnx.backend.test import __file__ as backend_folder


class OnnxBackendTest:
    """
    Definition of a backend test. It starts with a folder,
    in this folder, one onnx file must be there, then a subfolder
    for each test to run with this model.

    :param folder: test folder
    :param onnx_path: onnx file
    :param onnx_model: loaded onnx file
    :param tests: list of test
    """
    @staticmethod
    def _sort(filenames):
        temp = []
        for f in filenames:
            name = os.path.splitext(f)[0]
            i = name.split('_')[-1]
            temp.append((int(i), f))
        temp.sort()
        return [_[1] for _ in temp]

    @staticmethod
    def _load(folder, names):
        res = []
        for name in names:
            full = os.path.join(folder, name)
            new_tensor = onnx.TensorProto()
            with open(full, 'rb') as f:
                new_tensor.ParseFromString(f.read())
            try:
                t = to_array(new_tensor)
            except (ValueError, TypeError) as e:
                raise RuntimeError(
                    "Unexpected format for %r. This may be not a tensor."
                    "" % full) from e
            res.append(t)
        return res

    def __repr__(self):
        "usual"
        return "%s(%r)" % (self.__class__.__name__, self.folder)

    def __init__(self, folder):
        if not os.path.exists(folder):
            raise FileNotFoundError("Unable to find folder %r." % folder)
        content = os.listdir(folder)
        onx = [c for c in content if os.path.splitext(c)[-1] in {'.onnx'}]
        if len(onx) != 1:
            raise ValueError(
                "There is more than one onnx file in %r (%r)." % (
                    folder, onx))
        self.folder = folder
        self.onnx_path = os.path.join(folder, onx[0])
        self.onnx_model = onnx.load(self.onnx_path)

        self.tests = []
        for sub in content:
            full = os.path.join(folder, sub)
            if os.path.isdir(full):
                pb = [c for c in os.listdir(full)
                      if os.path.splitext(c)[-1] in {'.pb'}]
                inputs = OnnxBackendTest._sort(
                    c for c in pb if c.startswith('input_'))
                outputs = OnnxBackendTest._sort(
                    c for c in pb if c.startswith('output_'))

                try:
                    t = dict(
                        inputs=OnnxBackendTest._load(full, inputs),
                        outputs=OnnxBackendTest._load(full, outputs))
                except RuntimeError:
                    # No tensors
                    t = dict(inputs=inputs, outputs=outputs)
                self.tests.append(t)

    @property
    def name(self):
        "Returns the test name."
        return os.path.split(self.folder)[-1]

    def __len__(self):
        "Returns the number of tests."
        return len(self.tests)

    def run(self, load_fct, run_fct, index=None, decimal=5):
        """
        Executes a tests or all tests if index is None.
        The function crashes if the tests fails.

        :param load_fct: loading function, takes a loaded onnx graph,
            and returns an object
        :param run_fct: running function, takes the result of previous
            function, the inputs, and returns the outputs
        :param index: index of the test to run or all.
        """
        if index is None:
            for i in range(len(self)):
                self.run(load_fct, run_fct, index=i)
            return

        obj = load_fct(self.onnx_model)

        got = run_fct(obj, *self.tests[index]['inputs'])
        expected = self.tests[index]['outputs']
        if len(got) != len(expected):
            raise AssertionError(
                "Unexpected number of output (test %d, folder %r), "
                "got %r, expected %r." % (
                    index, self.folder, len(got), len(expected)))
        for i, (e, o) in enumerate(zip(expected, got)):
            try:
                assert_almost_equal(e, o)
            except AssertionError as ex:
                raise AssertionError(
                    "Output %d of test %d in folder %r failed." % (
                        i, index, self.folder)) from ex

    def to_python(self):
        """
        Returns a python code equivalent to the ONNX test.

        :return: code
        """
        from ..onnx_tools.onnx_export import export2onnx
        rows = []
        code = export2onnx(self.onnx_model)
        lines = code.split('\n')
        lines = [line for line in lines
                 if not line.strip().startswith('print') and
                 not line.strip().startswith('# ')]
        rows.append(textwrap.dedent("\n".join(lines)))
        rows.append("oinf = OnnxInference(onnx_model)")
        for test in self.tests:
            rows.append("xs = [")
            for inp in test['inputs']:
                rows.append(textwrap.indent(repr(inp) + ',',
                                            '    ' * 2))
            rows.append("]")
            rows.append("ys = [")
            for out in test['outputs']:
                rows.append(textwrap.indent(repr(out) + ',',
                                            '    ' * 2))
            rows.append("]")
            rows.append("feeds = {n: x for n, x in zip(oinf.input_names, xs)}")
            rows.append("got = oinf.run(feeds)")
            rows.append("goty = [got[k] for k in oinf.output_names]")
            rows.append("for y, gy in zip(ys, goty):")
            rows.append("    self.assertEqualArray(y, gy)")
            rows.append("")
        code = "\n".join(rows)
        final = "\n".join(["def %s(self):" % self.name,
                           textwrap.indent(code, '    ')])
        try:
            from pyquickhelper.pycode.code_helper import remove_extra_spaces_and_pep8
        except ImportError:
            return final
        return remove_extra_spaces_and_pep8(final)


def enumerate_onnx_tests(series, fct_filter=None):
    """
    Collects test from a sub folder of `onnx/backend/test`.
    Works as an enumerator to start processing them
    without waiting or storing too much of them.

    :param series: which subfolder to load
    :param fct_filter: function `lambda testname: boolean`
        to load or skip the test, None for all
    :return: list of @see cl OnnxBackendTest
    """
    root = os.path.dirname(backend_folder)
    sub = os.path.join(root, 'data', series)
    if not os.path.exists(sub):
        raise FileNotFoundError(
            "Unable to find series of tests in %r, subfolders:\n%s" % (
                root, "\n".join(os.listdir(root))))
    tests = os.listdir(sub)
    for t in tests:
        if fct_filter is not None and not fct_filter(t):
            continue
        folder = os.path.join(sub, t)
        content = os.listdir(folder)
        onx = [c for c in content if os.path.splitext(c)[-1] in {'.onnx'}]
        if len(onx) == 1:
            yield OnnxBackendTest(folder)
            continue
