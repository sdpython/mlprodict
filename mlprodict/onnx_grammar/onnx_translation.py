"""
@file
@brief One class which visits a syntax tree.
"""
import inspect
import ast
from textwrap import dedent
from .node_visitor_translator import CodeNodeVisitor


def translate_fct2onnx(fct, context=None, cpl=False,
                       context_cpl=None,
                       output_names=None):
    """
    Translates a function into :epkg:`ONNX`. The code it produces
    is using classes *OnnxAbs*, *OnnxAdd*, ...

    @param      fct             function to convert
    @param      context         context of the function to convert
                                something like
                                ``{'numpy.transpose': numpy.transpose}``
    @param      cpl             compile the function after it was
                                created
    @param      context_cpl     context used at compiling time
    @param      output_names    names of the output in the :epkg:`ONNX` graph
    @return                     code or compiled code

    .. exref::
        :title: Convert a function into ONNX code

        The following code parses a python function and returns
        another python function which produces an :epkg:`ONNX`
        graph if executed.

        .. runpython::
            :showcode:
            :process:

            import numpy
            from mlprodict.onnx_grammar import translate_fct2onnx

            def trs(x, y):
                z = x + numpy.transpose(y, axes=[1, 0])
                return x * z

            onnx_code = translate_fct2onnx(
                trs, context={'numpy.transpose': numpy.transpose})
            print(onnx_code)

    Next example is goes further and compile the outcome.

    .. exref::
        :title: Convert a function into ONNX code and run

        The following code parses a python function and returns
        another python function which produces an :epkg:`ONNX`
        graph if executed. The example executes the function,
        creates an :epkg:`ONNX` then uses @see cl OnnxInference
        to compute *predictions*. Finally it compares
        them to the original.

        .. runpython::
            :showcode:
            :process:

            import numpy
            from mlprodict.onnx_grammar import translate_fct2onnx
            from mlprodict.onnxrt import OnnxInference
            from skl2onnx.algebra.onnx_ops import (
                OnnxAdd, OnnxTranspose, OnnxMul, OnnxIdentity
            )

            ctx = {'OnnxAdd': OnnxAdd,
                   'OnnxTranspose': OnnxTranspose,
                   'OnnxMul': OnnxMul,
                   'OnnxIdentity': OnnxIdentity}

            def trs(x, y):
                z = x + numpy.transpose(y, axes=[1, 0])
                return x * z

            inputs = {'x': numpy.array([[1, 2]], dtype=numpy.float32),
                      'y': numpy.array([[-0.3, 0.4]], dtype=numpy.float32).T}

            original = trs(inputs['x'], inputs['y'])

            print('original output:', original)


            onnx_fct = translate_fct2onnx(
                trs, context={'numpy.transpose': numpy.transpose},
                cpl=True, context_cpl=ctx, output_names=['Z'])

            onnx_code = onnx_fct('x', 'y')
            print('ONNX code:', onnx_code)

            onnx_g = onnx_code.to_onnx(inputs)

            oinf = OnnxInference(onnx_g)
            res = oinf.run(inputs)

            print("ONNX inference:", res['Z'])
            print("ONNX graph:", onnx_g)

    The function to be converted may include python functions
    which must not be converted. In that case, their name
    must be prefixed by ``py_``. The execution of the function
    this one builds produces the following error::

        TypeError: Parameter to MergeFrom() must be instance of same class:
        expected onnx.TensorProto got onnx.AttributeProto.

    It indicates that constants in the code marges multiple types,
    usually floats and tensor of floats. Floats should be converted
    using the following function::

        def py_make_float_array(cst):
            return numpy.array([cst], dtype=numpy.float32)


    .. exref::
        :title: Convert kernel ExpSineSquared from scikit-learn

        This kernel is used by :epkg:`sklearn:gaussian_process:GaussianProcessRegressor`
        and its conversion is required to convert the model.
        The first step is to write a standalone function which
        relies on :epkg:`scipy` or :epkg:`numpy` and which produces
        the same results. The second step calls this function to
        produces the :epkg:`ONNX` graph.

        .. runpython::
            :showcode:
            :process:

        import numpy
        from scipy.spatial.distance import squareform, pdist
        from sklearn.gaussian_process.kernels import ExpSineSquared
        from mlprodict.onnx_grammar import translate_fct2onnx
        from mlprodict.onnxrt import OnnxInference
        from skl2onnx.algebra.onnx_ops import (
            OnnxAdd, OnnxSin, OnnxMul, OnnxIdentity, OnnxPow, OnnxDiv, OnnxExp
        )
        from skl2onnx.algebra.complex_functions import squareform_pdist as Onnxsquareform_pdist


        def squareform_pdist(X, metric='sqeuclidean'):
            return squareform(pdist(X, metric=metric))


        def py_make_float_array(cst):
            return numpy.array([cst], dtype=numpy.float32)


        def kernel_call_ynone(X, length_scale=1.2, periodicity=1.1, pi=3.141592653589793):
            dists = squareform_pdist(X, metric='euclidean')

            t_pi = py_make_float_array(pi)
            t_periodicity = py_make_float_array(periodicity)
            arg = dists / t_periodicity * t_pi

            sin_of_arg = numpy.sin(arg)

            t_2 = py_make_float_array(2)
            t__2 = py_make_float_array(-2)
            t_length_scale = py_make_float_array(length_scale)

            K = numpy.exp((sin_of_arg / t_length_scale) ** t_2 * t__2)
            return K


        kernel = ExpSineSquared(length_scale=1.2, periodicity=1.1)

        x = numpy.array([[1, 2], [3, 4]], dtype=float)

        exp = kernel(x, None)
        got = kernel_call_ynone(x)
        print("ExpSineSquared:")
        print(exp)
        print("numpy function:")
        print(got)

        # conversion to ONNX and execution
        context = {'numpy.sin': numpy.sin, 'numpy.exp': numpy.exp,
                   'squareform_pdist': 'squareform_pdist',
                   'py_make_float_array': py_make_float_array}

        ctx = {'OnnxAdd': OnnxAdd, 'OnnxPow': OnnxPow,
               'OnnxSin': OnnxSin, 'OnnxDiv': OnnxDiv,
               'OnnxMul': OnnxMul, 'OnnxIdentity': OnnxIdentity,
               'OnnxExp': OnnxExp,
               'Onnxsquareform_pdist': Onnxsquareform_pdist,
               'py_make_float_array': py_make_float_array}

        # converts the numpy function into an ONNX function
        fct_onnx = translate_fct2onnx(kernel_call_ynone, context=context,
                                 cpl=True, context_cpl=ctx,
                                 output_names=['Z'])

        onnx_model = fct_onnx('X')

        # calls the ONNX function to get the ONNX graph
        inputs = {'X': x.astype(numpy.float32)}
        onnx_g = onnx_model.to_onnx(inputs)

        oinf = OnnxInference(onnx_g)
        res = oinf.run(inputs)
        print("ONNX output:")
        print(res['Z'])
        print("Function code:")
        print(translate_fct2onnx(kernel_call_ynone, context=context,
                                 output_names=['Z']))

    """
    def compile_code(name, code, context=None):
        """
        Compiles a python function with the given
        context.

        @param      name        function name
        @param      code        python code
        @param      context     context used at compilation
        @return                 compiled function
        """
        if context is None:
            context = {}
        obj = compile(code, "", "exec")
        context_g = context.copy()
        context_l = context.copy()
        exec(obj, context_g, context_l)  # pylint: disable=W0122
        return context_l[name]

    code = inspect.getsource(fct)
    node = ast.parse(dedent(code))
    v = CodeNodeVisitor()
    v.visit(node)
    onnx_code = v.export(context=context,
                         output_names=output_names)
    if not cpl:
        return onnx_code
    return compile_code(fct.__name__, onnx_code, context_cpl)
