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
    """
    if optim == 'cdist':
        options = {model.__class__: {'optim': 'cdist'}}
    elif optim == 'nozipmap':
        options = {model.__class__: {'zipmap': False}}
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
    """
    if opts is None:
        return opts
    if opts == {model: {'optim': 'cdist'}}:
        return 'cdist'
    if opts == {model: {'zipmap': False}}:
        return 'nozipmap'
    return None
