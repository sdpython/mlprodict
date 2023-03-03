"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Any, Dict, List, Tuple
from onnx import FunctionProto
from onnx.helper import make_function, make_node, make_opsetid
from .numpyx_var import FUNCTION_DOMAIN


def get_function_implementation(
        domop: Tuple[str, str], node_inputs: List[str],
        node_outputs: List[str], opsets: Dict[str, int],
        **kwargs: Any) -> FunctionProto:
    """
    Returns a :epkg:`FunctionProto` for a specific proto.

    :param domop: domain, function
    :param node_inputs: list of input names
    :param node_outputs: list of output names
    :param opsets: available opsets
    :kwargs: any other parameters
    :return: FunctionProto
    """
    if domop[0] != FUNCTION_DOMAIN:
        raise ValueError(
            f"This function only considers function for domain "
            f"{FUNCTION_DOMAIN!r} not {domop[0]!r}.")
    if domop[1] == "CDist":
        return _get_cdist_implementation(
            node_inputs, node_outputs, opsets, **kwargs)
    raise ValueError(
        f"Unable to return an implementation of function {domop!r}.")


def _get_cdist_implementation(
        node_inputs: List[str], node_outputs: List[str],
        opsets: Dict[str, int], **kwargs: Any) -> FunctionProto:
    """
    Returns the CDist implementation as a function.
    """
    if set(kwargs) != {'metric'}:
        raise ValueError(
            f"kwargs={kwargs} must contain metric and only metric.")
    if opsets is not None and "com.microsoft" in opsets:
        node = make_node("CDist", ["xa", "xb"], ["z"],
                         domain="com.microsoft", metric=kwargs['metric'])
        return make_function(
            "numpyx", "CDist", ["xa", "xb"], ["z"], [node],
            [make_opsetid("com.microsoft", 1)], ["metric"])
    raise NotImplementedError(
        f"Not implementation for CDist and opsets={opsets}.")
