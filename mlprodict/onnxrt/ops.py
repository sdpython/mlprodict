"""
@file
@brief Loads runtime operator.
"""


def load_op(onnx_node, desc=None, options=None, variables=None, dtype=None):
    """
    Sets up a class for a specific ONNX operator.

    @param      onnx_node       :epkg:`onnx` node
    @param      desc            internal representation
    @param      options         runtime options
    @param      variables       registered variables created by previous operators
    @param      dtype           float computational type
    @return                     runtime class
    """
    if desc is None:
        raise ValueError(  # pragma: no cover
            "desc should not be None.")
    if options is None:
        provider = 'python'  # pragma: no cover
    else:
        provider = options.get('provider', 'python')
        if 'provider' in options:
            options = options.copy()
            del options['provider']
    if provider == 'python':
        from .ops_cpu import load_op as lo
        return lo(onnx_node, desc=desc, options=options)
    if provider == 'empty':
        from .ops_empty import load_op as lo
        return lo(onnx_node, desc=desc, options=options)
    if provider == 'onnxruntime2':
        from .ops_onnxruntime import load_op as lo
        return lo(onnx_node, desc=desc, options=options,  # pylint: disable=E1123
                  variables=variables, dtype=dtype)
    raise ValueError("Unable to handle provider '{}'.".format(provider))
