"""
@file
@brief Shortcut to *ops_shape*.
"""
from ._element_wise import shape_add, shape_mul, shape_div, shape_sub

shape_functions = {
    k: v for k, v in globals() if k.startswith("shape_")
}
