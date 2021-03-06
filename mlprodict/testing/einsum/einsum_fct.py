"""
@file
@brief Main functions decomposing einsum computation into
more simple functions.
"""
from itertools import permutations
import time
import math
import numpy
from .einsum_impl import decompose_einsum_equation, apply_einsum_sequence


_einsum_cache = {}


def enumerate_cached_einsum():
    """
    Enumerates all cached einsum function.
    """
    global _einsum_cache  # pylint: disable=W0603
    for k, v in _einsum_cache.items():
        yield k, v


class CachedEinsum:
    """
    Stores all the necessary information to cache the preprocessing
    of a an einsum equation.

    :param equation: numpy equation
    :param runtime: see :func:`einsum
        <mlprodict.testing.einsum.einsum_fct.einsum>`
    :param opset: ONNX opset
    :param optimize: finds the best letter permutation
    :param dtype: dtype
    :param verbose: displays progress information

    The class creates the following attributes:
    * `equation_` corresponding to the best equivalent equation
    * `graph_`: the corresponding graph returned by function
        :func:`decompose_einsum_equation
        <mlprodict.testing.einsum.einsum_impl.decompose_einsum_equation> `
    * `onnx_`: if a conversion to onnx is used, stores the onnx graph
    * `runtime_`: a function used by `__call__`, calls the runtime
    """

    def __init__(self, equation, runtime='batch_dot', opset=None,
                 optimize=False, dtype=numpy.float64, verbose=None):
        self.equation = equation
        self.runtime = runtime
        self.opset = opset
        self.optimize = optimize
        self.dtype = dtype
        self.verbose = verbose

    def __repr__(self):
        "usual"
        return "%s(%r, %r, %r, %r, %r)" % (
            self.__class__.__name__, self.equation, self.runtime,
            self.opset, self.optimize, self.dtype)

    def default_inputs(self, N=None):
        """
        Returns default inputs (reshaped numpy.arange + 0.7i).

        :param N: dimension (all dimension have the same size)

        If *N is None*, N is given a size depending on the number of letters
        to avoid spending too much time on optimization.
        """
        if N is None:
            letters = set(c for c in self.equation
                          if "a" <= c <= "z" or "A" <= c <= "Z")
            nn = math.factorial(len(letters))
            N = max(int(2 ** 11 / nn), 4)
            N = min(N, 15)
        inps = self.equation.split('->')[0].split(',')
        lens = [len(s) for s in inps]
        inputs = [numpy.arange(N ** d).reshape((N,) * d) for d in lens]
        inputs = [(i + 0.7 * ii).astype(self.dtype)
                  for ii, i in enumerate(inputs)]
        return inputs

    def build(self):
        """
        Preprocesses the equation builds whatever is necessary
        to compute the result of the einsum equation.
        """
        if not self.optimize and not hasattr(self, 'equation_'):
            self.equation_ = self.equation
        else:
            # loops over all permutations
            if self.equation.lower() != self.equation:
                raise RuntimeError(
                    "Only lower equation can be optimized, %r is not." % self.equation)
            letters = list(
                sorted(set(c for c in self.equation if "a" <= c <= "z")))
            possible = list(permutations(letters))
            possible.insert(0, letters)
            if self.verbose:
                from tqdm import tqdm
                subset = tqdm(possible)
            else:
                subset = possible
            best = []
            very_best = None
            inputs = None
            for perm in subset:
                replace = {d: c for c, d in zip(letters, perm)}
                eq = self.equation
                for k, v in replace.items():
                    eq = eq.replace(k, v.upper())
                eq = eq.lower()
                inst = CachedEinsum(eq, runtime=self.runtime, opset=self.opset,
                                    optimize=False, dtype=self.dtype)
                inst.build()
                if inputs is None:
                    inputs = inst.default_inputs()
                    inst(*inputs)
                ts = time.perf_counter()
                for _ in range(0, 10):
                    inst(*inputs)
                delta = time.perf_counter() - ts
                if len(best) < 10:
                    best.append((delta, eq))
                    best.sort()
                elif delta < best[-1][0]:
                    best[-1] = (delta, eq)
                    best.sort()
                if self.verbose and (
                        very_best is None or very_best != best[0][0]):
                    very_best = best[0][0]
                    subset.set_description("%1.2g best=%r" % best[0])
            self.optimized_ = best
            self.equation_ = best[0][1]
        self.build_runtime()

    def build_runtime(self):
        """
        Builds the runtime associated to the
        equation `self.equation_`.
        """
        self.graph_ = decompose_einsum_equation(
            self.equation_, strategy='numpy', clean=True)
        if self.runtime == 'batch_dot':
            self.runtime_ = lambda *inputs: apply_einsum_sequence(
                self.graph_, *inputs)
        elif self.runtime in ('python', 'onnxruntime1'):
            from ...onnxrt import OnnxInference
            n_inputs = len(self.graph_.metadata['lengths']) - 1
            input_names = ['X%d' % i for i in range(n_inputs)]
            self.onnx_names_ = input_names
            onx = self.graph_.to_onnx(
                'Y', *input_names, opset=self.opset, dtype=self.dtype)
            self.onnx_ = onx
            rt = ('python_compiled'
                  if self.runtime == 'python'
                  else self.runtime)
            self.oinf_ = OnnxInference(onx, runtime=rt)
            self.runtime_ = lambda *inputs: self.oinf_.run(
                {i: v for i, v in zip(self.onnx_names_, inputs)})['Y']
        else:
            raise ValueError(
                "Unexpected runtime %r." % self.runtime)

    def __call__(self, *inputs):
        """
        Calls the runtime `self.runtime_`.
        """
        if not hasattr(self, 'runtime_'):
            raise RuntimeError(
                "Method build_runtime was not called.")
        return self.runtime_(*inputs)

    @staticmethod
    def build_einsum(equation, runtime, opset, optimize,
                     dtype, verbose=None):
        """
        Creates an instance of *CachedEinsum*.
        """
        inst = CachedEinsum(equation, runtime=runtime, opset=opset,
                            optimize=optimize, dtype=dtype, verbose=verbose)
        inst.build()
        return inst


def einsum(equation, *inputs, optimize=False, runtime="batch_dot",
           cache=True, opset=None, verbose=None):
    """
    Proposes a new implementatino of :epkg:`numpy:einsum`.
    It does not allow expresion using `...` and expects
    a right member.

    :param equation: einsum equation
    :param inputs: inputs
    :param optimize: permutes all letters to find the best
        permutation
    :param runtime: runtime used to compute the results once the
        computation graph is produced (see below)
    :param cache: if True, the function stores the preprocessing
        done for a specific equation, the second call with the same
        equation is much faster
    :param opset: ONNX opset to use for some runtimes
    :param verbose: display progress if optimize is True
    :return: einsum result

    The available runtimes are:
    * `batch_dot`: the runtime is @see fn apply_einsum_sequence,
    * `python`: one ONNX graph executed with a python runtime,
    * `onnxruntime1`: one ONNX graph executed with :epkg:`onnxruntime`.

    The function works in two steps:
    * first step analyses the equation to produce a computation graph,
      this graph can also be converted into ONNX,
    * second step runs the graph whatever the graph is.

    The function works the same way as :epkg:`numpy:einsum`:

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum import einsum

        equation = "abc,cd->abd"

        m1 = numpy.random.randn(2, 2, 2)
        m2 = numpy.random.randn(2, 2)

        np = numpy.einsum(equation, m1, m2)
        print('numpy.einsum')
        print(np)

        print('mlprodict.testing.einsum')
        mp = einsum(equation, m1, m2)
        print(mp)

    In some case, the einsum implementation can be optimized by looping
    on possible permutation:

    .. runpython::
        :showcode:
        :process:

        import timeit
        import numpy
        from mlprodict.testing.einsum import einsum
        from mlprodict.testing.einsum.einsum_fct import enumerate_cached_einsum

        equation = "cab,cd->ad"

        m1 = numpy.random.randn(20, 20, 20)
        m2 = numpy.random.randn(20, 20)

        print('numpy.einsum',
              timeit.timeit('numpy.einsum(equation, m1, m2)',
                            number=200,
                            globals=globals()))

        einsum(equation, m1, m2)
        print('einsum',
              timeit.timeit('einsum(equation, m1, m2)',
                            number=200,
                            globals=globals()))

        einsum(equation, m1, m2, runtime='python')
        print('einsum-python',
              timeit.timeit('einsum(equation, m1, m2, runtime="python")',
                            number=200,
                            globals=globals()))

        einsum(equation, m1, m2, runtime='onnxruntime1')
        print('einsum-onnxruntime1',
              timeit.timeit('einsum(equation, m1, m2, runtime="onnxruntime1")',
                            number=200,
                            globals=globals()))

        einsum(equation, m1, m2, runtime='onnxruntime1', optimize=True, verbose=1)
        print('einsum-onnxruntime1',
              timeit.timeit('einsum(equation, m1, m2, runtime="onnxruntime1", optimize=True)',
                            number=200,
                            globals=globals()))

        print("list of cached einsum equations")
        for k, v in enumerate_cached_einsum():
            print(k, v.equation, v.equation_)

    The last example shows the time taken by every function:

    .. runpython::
        :showcode:
        :process:

        import os
        from pyquickhelper.pycode.profiling import profile
        import numpy
        from mlprodict.testing.einsum import einsum
        from mlprodict.testing.einsum.einsum_fct import enumerate_cached_einsum
        from mlprodict import __file__ as path

        root = os.path.dirname(path)

        equation = "cab,cd->ad"

        m1 = numpy.random.randn(200, 20, 20)
        m2 = numpy.random.randn(200, 20)

        def clean(txt):
            txt = txt.replace(root, "mlprodict")
            return "\n".join(txt.split("\n")[:30])

        def fct1():
            for i in range(100):
                einsum(equation, m1, m2, cache=False)

        print("Profile cache with default runtime.")
        res = profile(fct1)
        print(root)
        print(clean(res[1]))

        def fct2():
            for i in range(100):
                einsum(equation, m1, m2, cache=False, runtime='python')

        print("Profile cache with runtime='python'.")
        res = profile(fct2)
        print(root)
        print(clean(res[1]))


        def fct3():
            for i in range(100):
                einsum(equation, m1, m2, cache=True)

        einsum(equation, m1, m2, cache=True)
        print("Profile execution with default runtime.")
        res = profile(fct3)
        print(root)
        print(clean(res[1]))



        def fct4():
            for i in range(100):
                einsum(equation, m1, m2, cache=True, runtime='python')

        einsum(equation, m1, m2, cache=True, runtime='python')
        print("Profile execution with runtime='python'.")
        res = profile(fct4)
        print(root)
        print(clean(res[1]))


        def fct5():
            for i in range(100):
                einsum(equation, m1, m2, cache=True, runtime='onnxruntime1')

        einsum(equation, m1, m2, cache=True, runtime='onnxruntime1')
        print("Profile execution with runtime='onnxruntime1'.")
        res = profile(fct5)
        print(root)
        print(clean(res[1]))
    """
    global _einsum_cache  # pylint: disable=W0603
    if len(inputs) == 0:
        raise ValueError("No inputs found.")
    dtypes = set(i.dtype for i in inputs)
    if len(dtypes) != 1:
        raise ValueError(
            "All inputs do not have the same type (%r), "
            "all of them should be cast before called einsum."
            "" % dtypes)
    cached = None
    if cache:
        key = equation, runtime, opset, optimize, inputs[0].dtype
        cached = _einsum_cache.get(key, None)
    if cached is None:
        cached = CachedEinsum.build_einsum(
            equation, runtime, opset, optimize,
            inputs[0].dtype, verbose=verbose)
    else:
        cache = False
    if cache:
        _einsum_cache[key] = cached
    return cached(*inputs)
