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
        options = optim
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

        from mlprodict.tools.asv_options_helper import benchmark_version
        print(benchmark_version())
    """
    return [11]


def ir_version():
    """
    Returns the preferred `IR_VERSION
    <https://github.com/onnx/onnx/blob/master/docs/IR.md#onnx-versioning>`_.

    .. runpython::
        :showcode:

        from mlprodict.tools.asv_options_helper import ir_version
        print(ir_version())
    """
    return [6]


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
    """
    if benchmark:
        return ir_version()[-1]
    from onnx import IR_VERSION  # pylint: disable=W0611
    return IR_VERSION
