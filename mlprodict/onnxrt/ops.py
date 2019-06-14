"""
@file
@brief
"""


def load_op(onnx_node, desc=None, options=None):
    """
    Sets up a class for a specific ONNX operator.

    @param      onnx_node       :epkg:`onnx` node
    @param      desc            internal representation
    @param      options         runtime options
    @return                     runtime class
    """
    if desc is None:
        raise ValueError("desc should not be None.")
    if options is None:
        provider = 'CPU'
    else:
        provider = options.get('provider', 'CPU')
        if 'provider' in options:
            options = options.copy()
            del options['provider']
    if provider == 'CPU':
        from .ops_cpu import load_op as lo
        return lo(onnx_node, desc=desc, options=options)
    else:
        raise ValueError("Unable to handle provider '{}'.".format(provider))
