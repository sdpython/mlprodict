"""
@file
@brief One class which visits a syntax tree.
"""
import inspect
import ast
from textwrap import dedent
import numpy
from scipy.spatial.distance import squareform, pdist
from .node_visitor_translator import CodeNodeVisitor


def py_make_float_array(cst, op_version=None):
    """
    Creates an array with a single element
    from a constant.

    @param      cst         constant
    @param      op_version  unused
    @return                 array

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.onnx_grammar.onnx_translation import py_make_float_array
        print(py_make_float_array(5.5))
    """
    return numpy.array([cst], dtype=numpy.float32)


def py_pow(x, p, op_version=None):
    """
    Function for python operator ``**``.

    @param      x           float
    @param      p           power
    @param      op_version  unused
    @return                 :math:`x^p`
    """
    return x ** p


def py_mul(*x, op_version=None):
    """
    Function for python operator ``*``.

    @param      x           floats
    @param      op_version  unused
    @return                 `x*y`
    """
    if len(x) == 2:
        return x[0] * x[1]
    p = x[0]
    for y in x[1:]:
        p *= y
    return p


def py_opp(x, op_version=None):
    """
    Function for python unary operator ``-``.

    @param      x           floats
    @param      op_version  unused
    @return                 `-x`
    """
    return -x


def squareform_pdist(X, metric='sqeuclidean', op_version=None):
    """
    Replacements for `squareform
    <http://scipy.github.io/devdocs/generated/scipy.spatial.distance.squareform.html>`_
    and `pdist
    <http://scipy.github.io/devdocs/generated/scipy.spatial.distance.pdist.html>`_.
    """
    return squareform(pdist(X, metric=metric))


def get_default_context():
    """
    Returns a default context useful for most of the conversion
    from a function using :epkg:`numpy` into :epkg:`ONNX`.
    """
    context = {'py_pow': py_pow, 'py_make_float_array': py_make_float_array,
               'py_mul': py_mul, 'py_opp': py_opp,
               'cdist': 'cdist', 'squareform_pdist': 'squareform_pdist'}
    allow = set(('abs add ceil arccos arccosh arcsin arcsinh arctan arctanh ceil cos cosh divide'
                 'equal exp floor greater invert less log matmul maximum minimum mod'
                 'multiply power sign sin sinh sqrt square subtract tan tanh transpose').split())
    for k, v in numpy.__dict__.items():
        if k not in allow:
            continue
        context['numpy.%s' % k] = v
        context['np.%s' % k] = v
    return context


def get_default_context_cpl():
    """
    Returns a default useful context to compile the converter
    returned by @see fn translate_fct2onnx.
    """
    ctx = {'py_make_float_array': py_make_float_array,
           'py_pow': py_pow, 'py_mul': py_mul, 'py_opp': py_opp,
           'numpy': numpy}
    try:
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        from skl2onnx.algebra.complex_functions import onnx_cdist
        ctx['onnx_squareform_pdist'] = onnx_squareform_pdist
        ctx['onnx_cdist'] = onnx_cdist
    except ImportError:  # pragma: no cover
        # Too old version for skl2onnx.
        pass

    from skl2onnx.algebra import onnx_ops
    from skl2onnx.algebra.onnx_operator import OnnxOperator
    d = onnx_ops.__dict__
    for k, v in d.items():
        try:
            if k.startswith("Onnx") and issubclass(v, OnnxOperator):
                ctx[k] = v
        except TypeError as e:
            if inspect.isfunction(v):
                continue
            raise RuntimeError(  # pragma: no cover
                "Issue with {}={} (type={})".format(k, v, type(v))) from e
    return ctx


def translate_fct2onnx(fct, context=None, cpl=False,
                       context_cpl=None, output_names=None,
                       dtype=numpy.float32,
                       verbose=0, fLOG=None):
    """
    Translates a function into :epkg:`ONNX`. The code it produces
    is using classes *OnnxAbs*, *OnnxAdd*, ...

    @param      fct             function to convert
    @param      context         context of the function to convert
                                something like ``{'numpy.transpose': numpy.transpose}``,
                                if *context* is None, it receives a default value
                                returnd by @see fn get_default_context
    @param      cpl             compile the function after it was
                                created
    @param      context_cpl     context used at compiling time
                                if *context_cpl* is None, it receives a default value
                                returnd by @see fn get_default_context_cpl
    @param      output_names    names of the output in the :epkg:`ONNX` graph
    @param      dtype           :epkg:`numpy` float type used to produce the model
    @param      verbose         integer, display more information
    @param      fLOG            logging function
    @return                     code or compiled code

    .. exref::
        :title: Convert a function into ONNX code

        The following code parses a python function and returns
        another python function which produces an :epkg:`ONNX`
        graph if executed.

        .. runpython::
            :showcode:
            :warningout: DeprecationWarning
            :process:
            :store_in_file: fct2onnx2.py

            import numpy
            from mlprodict.onnx_grammar import translate_fct2onnx

            def trs(x, y):
                z = x + numpy.transpose(y, axes=[1, 0])
                return x * z

            onnx_code = translate_fct2onnx(
                trs, context={'numpy.transpose': numpy.transpose})
            print(onnx_code)

    Next example goes further and compile the outcome.

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
            :warningout: DeprecationWarning
            :process:
            :store_in_file: fct2onnx3.py

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

            onnx_code = onnx_fct('x', 'y', opset_version=12)
            print('ONNX code:', onnx_code)

            onnx_g = onnx_code.to_onnx(inputs, target_opset=12)

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

    The function replaces empty contexts by default values which
    covers many :epkg:`numpy` functions. The tutorial
    :ref:`l-onnx-tutorial` gives an example of how it can be used
    on a more complex function.
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
            context = {}  # pragma: no cover
        try:
            obj = compile(code, "", "exec")
        except SyntaxError as e:  # pragma: no cover
            raise SyntaxError("Unable to compile\n{}".format(code)) from e
        context_g = context.copy()
        context_l = context.copy()
        exec(obj, context_g, context_l)  # pylint: disable=W0122
        return context_l[name]

    if isinstance(fct, str):
        code = fct
    elif callable(fct):
        code = inspect.getsource(fct)
    else:
        raise TypeError(  # pragma: no cover
            "Unable to guess code from type {}.".format(type(fct)))
    node = ast.parse(dedent(code))
    v = CodeNodeVisitor()
    v.visit(node)
    if context is None:
        context = get_default_context()
    onnx_code = v.export(context=context,
                         output_names=output_names)
    if not cpl:
        return onnx_code
    if verbose > 0 and fLOG is not None:  # pragma: no cover
        fLOG('[translate_fct2onnx] python code')
        fLOG(code)
        fLOG('[translate_fct2onnx] ONNX code')
        fLOG(onnx_code)
    if context_cpl is None:
        context_cpl = get_default_context_cpl()
    if 'numpy' not in context_cpl:
        context_cpl = context_cpl.copy()
        context_cpl['numpy'] = numpy
    return compile_code(fct.__name__, onnx_code, context_cpl)
