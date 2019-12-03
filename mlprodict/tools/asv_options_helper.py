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

        for name in ['cdist', 'nozipmap', 'raw_score']:
            print(name, ':', expand_onnx_options(LogisticRegression, name))
    """
    if optim == 'cdist':
        options = {model.__class__: {'optim': 'cdist'}}
    elif optim == 'nozipmap':
        options = {model.__class__: {'zipmap': False}}
    elif optim == 'raw_score':
        options = {model.__class__: {'raw_score': True}}
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
    if opts == {model: {'raw_score': True}}:
        return 'raw_score'
    return None
