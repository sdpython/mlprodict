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
