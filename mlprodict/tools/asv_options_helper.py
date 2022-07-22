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
