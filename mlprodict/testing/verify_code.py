"""
@file
@brief Looks into the code and detects error
before finalizing the benchmark.
"""
import ast


class ImperfectPythonCode(RuntimeError):
    """
    Raised if the code shows errors.
    """
    pass


def verify_code(source, exc=True):
    """
    Verifies :epkg:`python` code.

    @param      source      source to look into
    @param      exc         raise an exception or return the list of
                            missing identifiers
    @return                 tuple(missing identifiers, @see cl CodeNodeVisitor)
    """
    node = ast.parse(source)
    v = CodeNodeVisitor()
    v.visit(node)
    assign = v._assign
    imports = v._imports
    names = v._names
    args = v._args
    known = {'super': None, 'ImportError': None}
    for kn in imports:
        known[kn[0]] = kn
    for kn in assign:
        known[kn[0]] = kn
    for kn in args:
        known[kn[0]] = kn
    issues = set()
    for name in names:
        if name[0] not in known:
            issues.add(name[0])
    if exc and len(issues) > 0:
        raise ImperfectPythonCode(
            "Unknown identifiers: {} in\n{}".format(
                ",".join(issues), source))
    return issues, v


class CodeNodeVisitor(ast.NodeVisitor):
    """
    Visits the code, implements verification rules.
    """

    def __init__(self):
        ast.NodeVisitor.__init__(self)
        self._rows = []
        self._indent = 0
        self._stack = []
        self._imports = []
        self._names = []
        self._alias = []
        self._assign = []
        self._args = []
        self._fits = []

    def push(self, row):
        """
        Pushes an element into a list.
        """
        self._rows.append(row)

    def generic_visit(self, node):
        """
        Overrides ``generic_visit`` to check it is not used.
        """
        raise AttributeError(
            "generic_visit_args should be used.")  # pragma: no cover

    def generic_visit_args(self, node, row):
        """
        Overrides ``generic_visit`` to keep track of the indentation
        and the node parent. The function will add field
        ``row["children"] = visited`` nodes from here.

        @param      node        node which needs to be visited
        @param      row         row (a dictionary)
        @return                 See ``ast.NodeVisitor.generic_visit``
        """
        self._indent += 1
        last = len(self._rows)
        res = ast.NodeVisitor.generic_visit(  # pylint: disable=E1111
            self, node)  # pylint: disable=E1111
        row["children"] = [
            _ for _ in self._rows[
                last:] if _["indent"] == self._indent]
        self._indent -= 1
        return res

    def visit(self, node):
        """
        Visits a node, a method must exist for every object class.
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            if method.startswith("visit_"):
                cont = {
                    "indent": self._indent,
                    "str": method[6:],
                    "node": node}
                self.push(cont)
                return self.generic_visit_args(node, cont)
            raise TypeError("unable to find a method: " +
                            method)  # pragma: no cover
        res = visitor(node)
        # print(method, CodeNodeVisitor.print_node(node))
        return res

    @staticmethod
    def print_node(node):
        """
        Debugging purpose.
        """
        r = []
        for att in ["s", "name", "str", "id", "body", "n",
                    "arg", "targets", "attr", "returns", "ctx"]:
            if att in node.__dict__:
                r.append("{0}={1}".format(att, str(node.__dict__[att])))
        return " ".join(r)

    def print_tree(self):  # pylint: disable=C0116
        """
        Displays the tree of instructions.

        @return     string
        """
        rows = []
        for r in self.Rows:
            rows.append(
                ("{0}{1}: {2}".format(
                    "    " *
                    r["indent"],
                    r["type"],
                    r["str"])))
        return "\n".join(rows)

    @property
    def Rows(self):
        """
        returns a list of dictionaries with all the elements of the code
        """
        return [_ for _ in self._rows if not _.get("remove", False)]

    def visit_Str(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Str",
            "str": node.s,
            "node": node,
            "value": node.s}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Name(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Name",
            "str": node.id,
            "node": node,
            "id": node.id,
            "ctx": node.ctx}
        self.push(cont)
        self._names.append((node.id, node))
        return self.generic_visit_args(node, cont)

    def visit_Expr(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Expr",
            "str": '',
            "node": node,
            "value": node.value}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_alias(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "alias",
            "str": "",
            "node": node,
            "name": node.name,
            "asname": node.asname}
        self.push(cont)
        self._alias.append((node.name, node.asname, node))
        return self.generic_visit_args(node, cont)

    def visit_Module(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Module",
            "str": "",
            "body": node.body,
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Import(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Import",
            "str": "",
            "names": node.names,
            "node": node}
        self.push(cont)
        for name in node.names:
            self._imports.append((name.name, name.asname, node))
        return self.generic_visit_args(node, cont)

    def visit_ImportFrom(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "ImportFrom",
            "str": "",
            "module": node.module,
            "names": node.names,
            "node": node}
        self.push(cont)
        for name in node.names:
            self._imports.append((name.name, name.asname, node.module, node))
        return self.generic_visit_args(node, cont)

    def visit_ClassDef(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "ClassDef",
            "str": "",
            "name": node.name,
            "body": node.body,
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_FunctionDef(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "FunctionDef", "str": node.name, "name": node.name, "body": node.body,
                "node": node, "returns": node.returns}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_arguments(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "arguments", "str": "",
                "node": node, "args": node.args}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_arg(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "arg", "str": node.arg,
                "node": node,
                "arg": node.arg, "annotation": node.annotation}
        self.push(cont)
        self._args.append((node.arg, node))
        return self.generic_visit_args(node, cont)

    def visit_Assign(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "Assign", "str": "", "node": node,
                "targets": node.targets, "value": node.value}
        self.push(cont)
        for t in node.targets:
            if hasattr(t, 'id'):
                self._assign.append((t.id, node))
            else:
                self._assign.append((id(t), node))
        return self.generic_visit_args(node, cont)

    def visit_Store(self, node):  # pylint: disable=C0116
        #cont = { "indent":self._indent, "type": "Store", "str": "" }
        # self.push(cont)
        cont = {}
        return self.generic_visit_args(node, cont)

    def visit_Call(self, node):  # pylint: disable=C0116
        if "attr" in node.func.__dict__:
            cont = {"indent": self._indent, "type": "Call", "str": node.func.attr,
                    "node": node, "func": node.func}
        elif "id" in node.func.__dict__:
            cont = {"indent": self._indent, "type": "Call", "str": node.func.id,
                    "node": node, "func": node.func}
        else:
            cont = {"indent": self._indent, "type": "Call", "str": "",  # pragma: no cover
                    "node": node, "func": node.func}
        self.push(cont)
        if cont['str'] == 'fit':
            self._fits.append(cont)
        return self.generic_visit_args(node, cont)

    def visit_Attribute(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "Attribute", "str": node.attr,
                "node": node, "value": node.value, "ctx": node.ctx, "attr": node.attr}
        self.push(cont)
        # last = len(self._rows)
        res = self.generic_visit_args(node, cont)

        if len(cont["children"]) > 0:
            fir = cont["children"][0]
            if 'type' in fir and fir["type"] == "Name":
                parent = fir["node"].id
                cont["str"] = "{0}.{1}".format(parent, cont["str"])
                cont["children"][0]["remove"] = True
        return res

    def visit_Load(self, node):  # pylint: disable=C0116
        cont = {}
        return self.generic_visit_args(node, cont)

    def visit_keyword(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "keyword", "str": "{0}".format(node.arg),
                "node": node, "arg": node.arg, "value": node.value}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_BinOp(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "BinOp",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_UnaryOp(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "UnaryOp",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Not(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Not",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Invert(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Invert",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_BoolOp(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "BoolOp",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Mult(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Mult",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Div(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Div",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_FloorDiv(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "FloorDiv",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Add(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Add",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Pow(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Pow",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_In(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "In",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_AugAssign(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "AugAssign",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Eq(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Eq",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_IsNot(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "IsNot",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Is(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Is",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_And(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "And",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_BitAnd(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "BitAnd",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Or(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Or",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_NotEq(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "NotEq",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Mod(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Mod",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Sub(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Sub",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_USub(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "USub",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Compare(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Compare",
            "str": "",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Gt(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "Gt", "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_GtE(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "GtE", "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Lt(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "Lt", "str": "", "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Num(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Num",
            "node": node,
            "str": "{0}".format(
                node.n),
            'n': node.n}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Return(self, node):  # pylint: disable=C0116
        cont = {"indent": self._indent, "type": "Return", "node": node, "str": "",
                'value': node.value}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_List(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "List",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_ListComp(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "ListComp",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_comprehension(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "comprehension",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Dict(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Dict",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Tuple(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "Tuple",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_NameConstant(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "type": "NameConstant",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_(self, node):  # pylint: disable=C0116
        raise RuntimeError(  # pragma: no cover
            "This node is not handled: {}".format(node))

    def visit_Subscript(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "str": "Subscript",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_ExtSlice(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "str": "ExtSlice",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Slice(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "str": "Slice",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Index(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "str": "Index",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_If(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "str": "If",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_IfExp(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "str": "IfExp",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_Lambda(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "str": "Lambda",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)

    def visit_GeneratorExp(self, node):  # pylint: disable=C0116
        cont = {
            "indent": self._indent,
            "str": "GeneratorExp",
            "node": node}
        self.push(cont)
        return self.generic_visit_args(node, cont)
