"""
@file
@brief Class ShapeContainer
"""
from .shape_result import ShapeResult


class ShapeContainer:
    """
    Stores all infered shapes.
    """

    def __init__(self):
        self.shapes = dict()

    def __len__(self):
        "usual"
        return len(self.shapes)

    def __getitem__(self, key):
        "Retrieves one shape from its name."
        return self.shapes[key]

    def copy(self):
        "Makes a copy."
        cont = ShapeContainer()
        cont.shapes = self.shapes.copy()
        return cont

    def update(self, key, value):
        """
        Updates one shape. Returns True if the shape was different.
        """
        if not isinstance(key, str):
            raise TypeError("key must be a string not %r." % type(key))
        if not isinstance(value, ShapeResult):
            raise TypeError("value must be a ShapeResult not %r." % type(key))
        if key not in self.shapes:
            self.shapes[key] = value
            return True
        if self.shapes[key] == value:
            return False
        self.shapes[key] = value
        return True

    def __contains__(self, key):
        "Operator in."
        return key in self.shapes

    def __str__(self):
        """
        Displays.
        """
        rows = ["ShapeContainer({"]
        for k, v in self.shapes.items():
            rows.append("    %s: %r" % (k, v))
        rows.append("})")
        return "\n".join(rows)
