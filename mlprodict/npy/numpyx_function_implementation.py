"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Any, Dict, List, Tuple
from onnx import AttributeProto, FunctionProto, ValueInfoProto
from onnx.helper import (
    make_function, make_graph, make_node, make_opsetid,
    make_tensor_value_info)
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
        node = make_node("CDist", ["xa", "xb"], ["z"], domain="com.microsoft")
        att = AttributeProto()
        att.name = "metric"
        att.ref_attr_name = "metric"
        att.type = AttributeProto.STRING
        node.attribute.append(att)
        return make_function(
            "numpyx", "CDist", ["xa", "xb"], ["z"], [node],
            [make_opsetid("com.microsoft", 1)], ["metric"])

    # constant
    cst = make_node("Constant", [], ["metric"])
    att = AttributeProto()
    att.name = "value_string"
    att.ref_attr_name = "metric"
    att.type = AttributeProto.STRING
    cst.attribute.append(att)
    le = make_node("LabelEncoder", ["metric"], ["metric_int"],
                   keys_strings=["euclidean"],
                   values_int64s=[1],
                   domain="ai.onnx.ml")
    cst1 = make_node("Constant", [], ["one"], value_int=1)
    eq = make_node("Equal", ["metric_int"], ["euclidean"])

    # subgraph
    nodes = [make_node("Sub", ["next", "next_in"], ["sub"]),
             make_node("Constant", [], ["axis"], value_floats=[1]),
             make_node("ReduceSumSquare", ["sub", "axis"], ["scan_out"]),
             make_node("Identity", ["next_in"], ["next_out"])
             ]

    def make_value(name):
        value = ValueInfoProto()
        value.name = name
        return value

    graph = make_graph(
        nodes, "loop",
        [make_value("next"), make_value("next_in")],
        [make_value("scan_out"), make_value("next_out")])

    scan = make_node(
        "Scan", ["xa", "xb"], ["z", "next_out"],
        num_scan_inputs=1, graph=graph)
    z = make_value("z")
    then_branch = make_graph([scan], "gr", [], [z])

    node = make_node("If", ["euclidean"], ["z"], then_branch=then_branch,
                     else_branch=then_branch)
    return make_function(
        "numpyx", "CDist", ["xa", "xb"], ["z"],
        [cst, le, cst1, eq, node],
        [make_opsetid("", opsets[""])], ["metric"])
