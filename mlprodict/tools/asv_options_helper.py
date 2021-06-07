"""
@file
@brief Functions to show shortened options in :epkg:`asv` benchmarks.
"""


def expand_onnx_options(model, optim):
    """
    Expands shortened options. Long names hide some part
    of graphs in :epkg:`asv` benchmark. This trick converts
    a string into real conversions options.

    @param      model       model class (:epkg:`scikit-learn`)
    @param      optim       option
    @return                 expanded options

    It is the reverse of function @see fn shorten_onnx_options.
    The following options are handled:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from sklearn.linear_model import LogisticRegression
        from mlprodict.tools.asv_options_helper import expand_onnx_options

        for name in ['cdist', 'nozipmap', 'raw_scores']:
            print(name, ':', expand_onnx_options(LogisticRegression, name))
    """
    if optim == 'cdist':
        options = {model.__class__: {'optim': 'cdist'}}
    elif optim == 'nozipmap':
        options = {model.__class__: {'zipmap': False}}
    elif optim == 'raw_scores':
        options = {model.__class__: {'raw_scores': True, 'zipmap': False}}
    else:
        options = optim  # pragma: no cover
    return options


def shorten_onnx_options(model, opts):
    """
    Shortens onnx options into a string.
    Long names hide some part
    of graphs in :epkg:`asv` benchmark.

    @param      model       model class (:epkg:`scikit-learn`)
    @param      opts        options
    @return                 shortened options

    It is the reverse of function @see fn expand_onnx_options.
    """
    if opts is None:
        return opts
    if opts == {model: {'optim': 'cdist'}}:
        return 'cdist'
    if opts == {model: {'zipmap': False}}:
        return 'nozipmap'
    if opts == {model: {'raw_scores': True, 'zipmap': False}}:
        return 'raw_scores'
    return None


def benchmark_version():
    """
    Returns the list of ONNX version to benchmarks.
    Following snippet of code shows which version is
    current done.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.tools.asv_options_helper import benchmark_version
        print(benchmark_version())
    """
    return [14]  # opset=13, 14, ...


def ir_version():
    """
    Returns the preferred `IR_VERSION
    <https://github.com/onnx/onnx/blob/master/docs/IR.md#onnx-versioning>`_.

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.tools.asv_options_helper import ir_version
        print(ir_version())
    """
    return [7]


def get_opset_number_from_onnx(benchmark=True):
    """
    Retuns the current :epkg:`onnx` opset
    based on the installed version of :epkg:`onnx`.

    @param      benchmark       returns the latest
                                version usable for benchmark
    @eturn                      opset number
    """
    if benchmark:
        return benchmark_version()[-1]
    from onnx.defs import onnx_opset_version  # pylint: disable=W0611
    return onnx_opset_version()


def get_ir_version_from_onnx(benchmark=True):
    """
    Retuns the current :epkg:`onnx` :epkg:`IR_VERSION`
    based on the installed version of :epkg:`onnx`.

    @param      benchmark       returns the latest
                                version usable for benchmark
    @eturn                      opset number

    .. faqref::
        :title: Failed to load model with error: Unknown model file format version.
        :lid: l-onnx-ir-version-fail

        :epkg:`onnxruntime` (or ``runtime='onnxruntime1'`` with @see cl OnnxInference)
        fails sometimes to load a model showing the following error messsage:

        ::

            RuntimeError: Unable to create InferenceSession due to '[ONNXRuntimeError] :
            2 : INVALID_ARGUMENT : Failed to load model with error: Unknown model file format version.'

        This case is due to metadata ``ir_version`` which defines the
        :epkg:`IR_VERSION` or *ONNX version*. When a model is machine learned
        model is converted, it is usually done with the default version
        (``ir_version``) returned by the :epkg:`onnx` package.
        :epkg:`onnxruntime` raises the above mentioned error message
        when this version (``ir_version``) is too recent. In this case,
        :epkg:`onnxruntime` should be updated to the latest version
        available or the metadata ``ir_version`` can just be changed to
        a lower number. Th function @see fn get_ir_version_from_onnx
        returns the latest tested version with *mlprodict*.

        .. runpython::
            :showcode:
            :warningout: DeprecationWarning

            from sklearn.linear_model import LinearRegression
            from sklearn.datasets import load_iris
            from mlprodict.onnxrt import OnnxInference
            import numpy

            iris = load_iris()
            X = iris.data[:, :2]
            y = iris.target
            lr = LinearRegression()
            lr.fit(X, y)

            # Conversion into ONNX.
            from mlprodict.onnx_conv import to_onnx
            model_onnx = to_onnx(lr, X.astype(numpy.float32),
                                 target_opset=12)
            print("ir_version", model_onnx.ir_version)

            # Change ir_version
            model_onnx.ir_version = 6

            # Predictions with onnxruntime
            oinf = OnnxInference(model_onnx, runtime='onnxruntime1')
            ypred = oinf.run({'X': X[:5].astype(numpy.float32)})
            print("ONNX output:", ypred)

            # To avoid keep a fixed version number, you can use
            # the value returned by function get_ir_version_from_onnx
            from mlprodict.tools import get_ir_version_from_onnx
            model_onnx.ir_version = get_ir_version_from_onnx()
            print("ir_version", model_onnx.ir_version)
    """
    if benchmark:
        return ir_version()[-1]
    from onnx import IR_VERSION  # pylint: disable=W0611
    return IR_VERSION


def display_onnx(model_onnx, max_length=1000):
    """
    Returns a shortened string of the model.

    @param      model_onnx      onnx model
    @param      max_length      maximal string length
    @return                     string
    """
    res = str(model_onnx)
    if max_length is None or len(res) <= max_length:
        return res
    begin = res[:max_length // 2]
    end = res[-max_length // 2:]
    return "\n".join([begin, '[...]', end])


def version2number(vers):
    """
    Converts a version number into a real number.
    """
    spl = vers.split('.')
    r = 0
    for i, s in enumerate(spl):
        try:
            vi = int(s)
        except ValueError:
            vi = 0
        r += vi * 10 ** (-i * 3)
    return r
