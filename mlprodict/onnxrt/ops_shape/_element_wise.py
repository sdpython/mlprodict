"""
@file
@brief Computes shape inference for element wise operators.
"""

def _element_wise(known_shapes, node, x, y):
    """
    Infers shape for an element wise operator.

    :param known_shapes: known shapes
    :param node: Onnx node
    :param x: first argument
    :param y: second argument
    :return: 
    """
    raise NotImplementedError()
    

def shape_add(known_shapes, node, x, y):
    "Infers shape for operator Add."
    return _element_wise(known_shapes, node, x, y)
    
def shape_sub(known_shapes, node, x, y):
    "Infers shape for operator Sub."
    return _element_wise(known_shapes, node, x, y)
    
def shape_div(known_shapes, node, x, y):
    "Infers shape for operator Div."
    return _element_wise(known_shapes, node, x, y)
    
def shape_mul(known_shapes, node, x, y):
    "Infers shape for operator Mul."
    return _element_wise(known_shapes, node, x, y)
