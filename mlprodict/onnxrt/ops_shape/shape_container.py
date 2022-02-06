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
        self.names = dict()
        self.names_rev = dict()

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
        cont.names = self.names.copy()
        cont.names_rev = {k: v.copy() for k, v in self.names_rev.items()}
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

    def get_new_name(self, name, result_name, dim):
        """
        Returns a variable name when a dimension is not
        specified.
        """
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be string not %r." % name)
        if name is None:
            name = ''
        if name == '' or name not in self.names:
            i = 0
            new_name = "%s_%d" % (name, i)
            while new_name in self.names:
                i += 1
                new_name = "%s_%d" % (name, i)
            self.names[new_name] = (name, result_name, dim)
            if name not in self.names_rev:
                self.names_rev[name] = []
            self.names_rev[name].append(new_name)
            return new_name
        val = self.names_rev[name]
        if len(val) != 1:
            raise RuntimeError(
                "Name %r has more than one correspondance (%r)." % (
                    name, val))
        return val[0]

    def get_all_constraints(self):
        """
        Gathers all constraintes while inferring the shape.
        """
        cons = {}
        for _, v in self.shapes.items():
            if v.constraints is not None:
                for c in v.constraints:
                    if c.name not in cons:
                        cons[c.name] = []
                    cons[c.name].append(c)
        for _, v in cons.items():
            if len(v) > 1:
                v[0].merge(v[1:])
            del v[1:]
        return cons
