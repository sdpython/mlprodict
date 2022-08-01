"""
@file
@brief Class ShapeContainer
"""
import pprint
from .shape_result import ShapeResult


class ShapeContainer:
    """
    Stores all infered shapes as @see cl ShapeResult.

    Attributes:

    * `shapes`: dictionary `{ result name: ShapeResult }`
    * `names`: some dimensions are unknown and represented as
        variables, this dictionary keeps track of them
    * `names_rev`: reverse dictionary of `names`
    """

    def __init__(self):
        self.shapes = dict()
        self.names = dict()
        self.names_rev = dict()

    def __repr__(self):
        "usual"
        return f"{self.__class__.__name__}()"

    def __len__(self):
        "usual"
        return len(self.shapes)

    def __getitem__(self, key):
        "Retrieves one shape from its name."
        return self.shapes[key]

    def copy(self, deep=False):
        "Makes a copy."
        cont = ShapeContainer()
        cont.shapes = {k: v.copy(deep=deep) for k, v in self.shapes.items()}
        cont.names = self.names.copy()
        cont.names_rev = {k: v.copy() for k, v in self.names_rev.items()}
        return cont

    def update(self, key, value):
        """
        Updates one shape. Returns True if the shape was different.
        """
        if not isinstance(key, str):
            raise TypeError(  # pragma: no cover
                f"key must be a string not {type(key)!r}.")
        if not isinstance(value, ShapeResult):
            raise TypeError(  # pragma: no cover
                f"value must be a ShapeResult not {type(key)!r}.")
        if key not in self.shapes:
            self.shapes[key] = value
            return True
        r = self.shapes[key].merge(value)
        return r

    def __contains__(self, key):
        "Operator in."
        return key in self.shapes

    def __str__(self):
        """
        Displays.
        """
        rows = ["ShapeContainer({"]
        for k, v in self.shapes.items():
            rows.append(f"    {k!r}: {v!r}")
        rows.append("}, names={")
        for k, v in self.names.items():
            rows.append(f"    {k!r}: {v!r}")
        cst = self.get_all_constraints()
        if len(cst) > 0:
            rows.append("}, constraint={")
            for c, v in cst.items():
                rows.append(f"    {c!r}: {v!r}")
            rows.append("})")
        else:
            rows.append("})")

        return "\n".join(rows)

    def get_new_name(self, name, result_name, dim):
        """
        Returns a variable name when a dimension is not
        specified.
        """
        if name is not None and not isinstance(name, str):
            raise TypeError(  # pragma: no cover
                f"name must be string not {name!r}.")
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
            raise RuntimeError(  # pragma: no cover
                f"Name {name!r} has more than one correspondance ({val!r}).")
        return val[0]

    def get_all_constraints(self):
        """
        Gathers all constraints.
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

    def get(self):
        """
        Returns the value of attribute `resolved_`
        (method `resolve()` must have been called first).
        """
        if not hasattr(self, 'resolved_') or self.resolved_ is None:
            raise AttributeError(  # pragma: no cover
                "Attribute 'resolved_' is missing. You must run "
                "method 'resolve()'.")
        return self.resolved_

    def resolve(self):
        """
        Resolves all constraints. It adds the attribute
        `resolved_`.
        """
        def vars_in_values(values):
            i_vals, s_vals = [], []
            for v in values:
                if isinstance(v, str):
                    s_vals.append(v)
                else:
                    i_vals.append(v)
            return set(i_vals), s_vals

        variables = {}
        for _, v in self.shapes.items():
            for sh in v.shape:
                if isinstance(sh, str):
                    variables[sh] = None

        # first step: resolves all constraint with integer
        dcsts = self.get_all_constraints()
        csts = []
        for li in dcsts.values():
            csts.extend(li)
        new_csts = []
        for cst in csts:
            if cst.name in variables and variables[cst.name] is None:
                if all(map(lambda n: isinstance(n, int), cst.values)):
                    variables[cst.name] = cst.values.copy()
                else:
                    new_csts.append(cst)
            else:
                raise RuntimeError(  # pragma: no cover
                    "Unable to find any correspondance for variable %r "
                    "in %r." % (cst.name, ", ".join(sorted(variables))))

        # second step: everything else, like a logic algorithm
        dim_names = set()
        csts = new_csts
        updates = 1
        while updates > 0 and len(new_csts) > 0:
            updates = 0
            new_csts = []
            for cst in csts:
                rvalues = variables[cst.name]
                ivalues, lvars = vars_in_values(cst.values)

                if len(lvars) > 0:
                    miss = 0
                    for lv in lvars:
                        if lv in variables and variables[lv] is not None:
                            ivalues |= variables[lv]
                        else:
                            miss += 1

                if miss == 0:
                    # simple case: only integers
                    if rvalues is None:
                        inter = ivalues
                    else:
                        inter = rvalues.intersection(ivalues)
                    if len(inter) == 0:
                        raise RuntimeError(  # pragma: no cover
                            "Resolution failed for variable %r, "
                            "current possibilities %r does not match "
                            "constraint %r." % (cst.name, rvalues, cst))
                    if rvalues is None or len(inter) < len(rvalues):
                        variables[cst.name] = inter
                        updates += 1
                    else:
                        continue
                elif len(dim_names) > 0:
                    # more complex case: variables
                    if len(cst.values) == 1 and len(lvars) == 1:
                        # exact mapping between cst.name and lvars[0]
                        a, b = cst.name, lvars[0]
                        if variables[a] is None and variables[b] is not None:
                            if variables[b].intersection(dim_names):
                                variables[a] = variables[b]
                                updates += 1
                                continue
                        elif variables[b] is None and variables[a] is not None:
                            if variables[a].intersection(dim_names):
                                variables[b] = variables[a]
                                updates += 1
                                continue

                new_csts.append(cst)
                csts = new_csts

            if len(new_csts) > 0 and updates == 0:
                # It means that a dimension needs to be left unknown.
                found = None
                for k, v in variables.items():
                    if v is None:
                        found = k
                if found is not None:
                    name = f"d{len(dim_names)}"
                    dim_names.add(name)
                    variables[found] = {name}
                    updates += 1
                else:
                    raise RuntimeError(  # pragma: no cover
                        f"Inconsistency in {self!r} with\n{variables!r}")

        # final
        results = {}
        for k, v in self.shapes.items():
            try:
                results[k] = v.resolve(variables)
            except RuntimeError as e:  # pragma: no cover
                raise RuntimeError(
                    "Unable to resolve shapes and constraints:\n%s"
                    "" % pprint.pformat(self.shapes)) from e
        self.resolved_ = results
        return self.resolved_
