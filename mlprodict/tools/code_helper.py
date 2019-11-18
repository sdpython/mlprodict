"""
@file
@brief A couple of tools unrelated to what the package does.
"""
import keyword
import re


def change_style(name):
    """
    Switches from *AaBb* into *aa_bb*.

    @param      name    name to convert
    @return             converted name
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return s2 if not keyword.iskeyword(s2) else s2 + "_"
