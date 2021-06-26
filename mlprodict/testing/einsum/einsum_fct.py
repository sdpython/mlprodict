"""
@file
@brief Main functions decomposing einsum computation into
more simple functions.
"""
from itertools import permutations
import time
import math
import numpy
from onnx import helper
from skl2onnx.common.data_types import FloatTensorType
from ...onnx_tools.onnx2py_helper import guess_proto_dtype
from ...tools.onnx_micro_runtime import OnnxMicroRuntime
from ...tools.asv_options_helper import (
    get_opset_number_from_onnx, get_ir_version_from_onnx)
from .einsum_impl import decompose_einsum_equation, apply_einsum_sequence
from .einsum_ml import predict_transposition_cost


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
    :param decompose: to decompose Einsum operator or to keep it as is
    :param key: key used to cache this class
    :param strategy: optimization strategy
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
                 optimize=False, dtype=numpy.float64, decompose=True,
                 strategy=None, verbose=None, key=None):
        self.equation = equation
        self.runtime = runtime
        self.opset = opset
        self.optimize = optimize
        self.dtype = dtype
        self.decompose = decompose
        self.strategy = strategy
        self.verbose = verbose
        self.key = key

    def __repr__(self):
        "usual"
        return "%s(%r, %r, %r, %r, %r, %r, %r, key=%r)" % (
            self.__class__.__name__, self.equation, self.runtime,
            self.opset, self.optimize, self.dtype, self.decompose,
            self.strategy, self.key)

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
        elif self.strategy is None:
            self.equation_ = self._build_optimize()
        elif self.strategy == 'ml':
            self.equation_ = self._build_optimize_ml()
        else:
            raise ValueError(  # pragma error
                "Unknown strategy %r." % self.strategy)
        self.build_runtime()

    def _build_optimize(self):
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
        confs = []
        very_best = None
        inputs = None
        for perm in subset:
            replace = {d: c for c, d in zip(letters, perm)}
            eq = self.equation
            for k, v in replace.items():
                eq = eq.replace(k, v.upper())
            eq = eq.lower()
            inst = CachedEinsum(eq, runtime=self.runtime, opset=self.opset,
                                optimize=False, dtype=self.dtype,
                                decompose=self.decompose)
            inst.build()
            if inputs is None:
                inputs = inst.default_inputs()
                inst(*inputs)
            ts = time.perf_counter()
            for _ in range(0, 10):
                inst(*inputs)
            delta = time.perf_counter() - ts
            confs.append((delta, eq))
            if len(best) < 10:
                best.append((delta, eq))
                best.sort()
            elif delta < best[-1][0]:
                best[-1] = (delta, eq)
                best.sort()
            if self.verbose and (
                    very_best is None or very_best != best[0][0]):
                very_best = best[0][0]
                subset.set_description("%1.2g rtbest=%r" % best[0])
        self.optimized_ = best
        self.timed_permutations_ = confs
        return best[0][1]

    def _build_optimize_ml(self):
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
        confs = []
        very_best = None
        inputs = None
        for perm in subset:
            replace = {d: c for c, d in zip(letters, perm)}
            eq = self.equation
            for k, v in replace.items():
                eq = eq.replace(k, v.upper())
            eq = eq.lower()
            inst = CachedEinsum(eq, runtime=self.runtime, opset=self.opset,
                                optimize=False, dtype=self.dtype,
                                decompose=self.decompose)
            inst.build()
            if inputs is None:
                inputs = inst.default_inputs()
            if hasattr(inst, 'onnx_'):
                onx = inst.onnx_
            else:
                inits = [
                    ('X%d' % i, FloatTensorType(list(inputs[i].shape)))
                    for i in range(len(inputs))]
                onx = inst.graph_.to_onnx('Y', *inits, opset=self.opset)

            rt = OnnxMicroRuntime(onx)
            dict_inputs = {'X%d' % i: inp for i, inp in enumerate(inputs)}
            out = rt.run(dict_inputs)

            transposes = []
            for node in onx.graph.node:  # pylint: disable=E1101
                if node.op_type == 'Transpose':
                    shape = [(d * 10 if d > 1 else d)
                             for d in out[node.input[0]].shape]
                    transposes.append(
                        [shape, list(node.attribute[0].ints)])

            delta = sum(max(0, predict_transposition_cost(*v))
                        for v in transposes)

            confs.append((delta, eq))
            if len(best) < 10:
                best.append((delta, eq))
                best.sort()
            elif delta < best[-1][0]:
                best[-1] = (delta, eq)
                best.sort()
            if self.verbose and (
                    very_best is None or very_best != best[0][0]):
                very_best = best[0][0]
                subset.set_description("%1.2g mlbest=%r" % best[0])
        self.optimized_ = best
        self.timed_permutations_ = confs
        return best[0][1]

    def build_onnx_einsum(self, input_names):
        """
        Builds an ONNX graph with a single einsum operator.
        """
        opset = (self.opset if self.opset is not None
                 else get_opset_number_from_onnx())
        ir_version = get_ir_version_from_onnx()
        proto_type = guess_proto_dtype(
            numpy.float32 if self.dtype is None else self.dtype)

        model = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=ir_version,
            producer_name='mlprodict',
            producer_version='0.0.1',
            graph=helper.make_graph(
                name='einsum',
                inputs=[helper.make_tensor_value_info(n, proto_type, None)
                        for n in input_names],
                outputs=[helper.make_tensor_value_info("Y", proto_type, None)],
                nodes=[
                    helper.make_node(
                        'Einsum', input_names, ["Y"], equation=self.equation_)]))
        return model

    def build_runtime(self):
        """
        Builds the runtime associated to the
        equation `self.equation_`.
        """
        if self.decompose:
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
                self.oinf_ = OnnxInference(self.onnx_, runtime=rt)
                self.runtime_ = lambda *inputs: self.oinf_.run(
                    {i: v for i, v in zip(self.onnx_names_, inputs)})['Y']
            else:
                raise ValueError(
                    "Unexpected runtime %r." % self.runtime)
        else:
            if self.runtime in ('python', 'onnxruntime1'):
                from ...onnxrt import OnnxInference
                n_inputs = len(self.equation.split('->')[0].split(','))
                input_names = ['X%d' % i for i in range(n_inputs)]
                self.onnx_ = self.build_onnx_einsum(input_names)
                self.onnx_names_ = input_names
                rt = ('python_compiled'
                      if self.runtime == 'python'
                      else self.runtime)
                self.oinf_ = OnnxInference(self.onnx_, runtime=rt)
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
                     dtype, decompose=True, strategy=None,
                     verbose=None, key=None):
        """
        Creates an instance of *CachedEinsum*.
        """
        inst = CachedEinsum(equation, runtime=runtime, opset=opset,
                            optimize=optimize, dtype=dtype,
                            decompose=decompose, strategy=strategy,
                            verbose=verbose, key=key)
        inst.build()
        return inst


def _einsum(equation, dtype, optimize=False, runtime="batch_dot",
            cache=True, opset=None, decompose=True, strategy=None,
            verbose=None):
    global _einsum_cache  # pylint: disable=W0603
    cached = None
    if cache:
        key = equation, runtime, opset, optimize, dtype, decompose, strategy
        cached = _einsum_cache.get(key, None)
    if cached is None:
        cached = CachedEinsum.build_einsum(
            equation, runtime, opset, optimize,
            dtype, decompose=decompose, strategy=strategy,
            verbose=verbose, key=key)
    else:
        cache = False
    if cache:
        _einsum_cache[key] = cached
    return cached


def einsum(equation, *inputs, optimize=False, runtime="batch_dot",
           cache=True, opset=None, decompose=True,
           strategy=None, verbose=None):
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
    :param decompose: by default, the function decomposes
        the equation into more simple operators but it can keep
        the original ONNX einsum operator.
    :param strategy: optimisation strategy (see below)
    :param verbose: display progress if optimize is True
    :return: einsum result

    The available runtimes are:
    * `batch_dot`: the runtime is @see fn apply_einsum_sequence,
    * `python`: one ONNX graph executed with a python runtime,
    * `onnxruntime1`: one ONNX graph executed with :epkg:`onnxruntime`.

    The optimisation strategy can be:
    * `None`: the same runtime is used to find the best permutation of letters
    * `'ml'`: a machine learned model is used to predict the
        best permutation of letters, this model comes from
        notebook :ref:`onnxoperatorcostrst`.

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
            return "\\n".join(txt.split("\\n")[:30])

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
    if len(inputs) == 0:
        raise ValueError("No inputs found.")
    dtypes = set(i.dtype for i in inputs)
    if len(dtypes) != 1:
        raise ValueError(
            "All inputs do not have the same type (%r), "
            "all of them should be cast before called einsum."
            "" % dtypes)
    cached = _einsum(equation, inputs[0].dtype, optimize=optimize,
                     runtime=runtime, cache=cache, opset=opset,
                     decompose=decompose, strategy=strategy, verbose=verbose)
    return cached(*inputs)
