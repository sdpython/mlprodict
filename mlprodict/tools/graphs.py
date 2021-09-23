"""
@file
@brief Alternative to dot to display a graph.

.. versionadded:: 0.7
"""
import pprint
import hashlib
import numpy
import onnx


def make_hash_bytes(data, length=20):
    """
    Creates a hash of length *length*.
    """
    m = hashlib.sha256()
    m.update(data)
    res = m.hexdigest()[:length]
    return res


class AdjacencyGraphDisplay:
    """
    Structure which contains the necessary information to
    display a graph using an adjacency matrix.

    .. versionadded:: 0.7
    """

    class Action:
        "One action to do."

        def __init__(self, x, y, kind, label, orientation=None):
            self.x = x
            self.y = y
            self.kind = kind
            self.label = label
            self.orientation = orientation

        def __repr__(self):
            "usual"
            return "%s(%r, %r, %r, %r, %r)" % (
                self.__class__.__name__,
                self.x, self.y, self.kind, self.label,
                self.orientation)

    def __init__(self):
        self.actions = []

    def __iter__(self):
        "Iterates over actions."
        for act in self.actions:
            yield act

    def __str__(self):
        "usual"
        rows = ["%s(" % self.__class__.__name__]
        for act in self:
            rows.append("    %r" % act)
        rows.append(")")
        return "\n".join(rows)

    def add(self, x, y, kind, label, orientation=None):
        """
        Adds an action to display the graph.

        :param x: x coordinate
        :param y: y coordinate
        :param kind: `'cross'` or `'text'`
        :param label: specific to kind
        :param orientation: a 2-uple `(i,j)` where *i* or *j* in `{-1,0,1}`
        """
        if kind not in {'cross', 'text'}:
            raise ValueError(  # pragma: no cover
                "Unexpected value for kind %r." % kind)
        if kind == 'cross' and label[0] not in {'I', 'O'}:
            raise ValueError(  # pragma: no cover
                "kind=='cross' and label[0]=%r not in {'I','O'}." % label)
        if not isinstance(label, str):
            raise TypeError(  # pragma: no cover
                "Unexpected label type %r." % type(label))
        self.actions.append(
            AdjacencyGraphDisplay.Action(x, y, kind, label=label,
                                         orientation=orientation))

    def to_text(self):
        """
        Displays the graph as a single string.
        See @see fn onnx2bigraph to see how the result
        looks like.

        :return: str
        """
        mat = {}
        for act in self:
            if act.kind == 'cross':
                if act.orientation != (1, 0):
                    raise NotImplementedError(  # pragma: no cover
                        "Orientation for 'cross' must be (1, 0) not %r."
                        "" % act.orientation)
                if len(act.label) == 1:
                    mat[act.x * 3, act.y] = act.label
                elif len(act.label) == 2:
                    mat[act.x * 3, act.y] = act.label[0]
                    mat[act.x * 3 + 1, act.y] = act.label[1]
                else:
                    raise NotImplementedError(
                        "Unable to display long cross label (%r)."
                        "" % act.label)
            elif act.kind == 'text':
                x = act.x * 3
                y = act.y
                orient = act.orientation
                charset = list(act.label if max(orient) == 1
                               else reversed(act.label))
                for c in charset:
                    mat[x, y] = c
                    x += orient[0]
                    y += orient[1]
            else:
                raise ValueError(  # pragma: no cover
                    "Unexpected kind value %r." % act.kind)

        min_i = min(k[0] for k in mat)
        min_j = min(k[1] for k in mat)
        mat2 = {}
        for k, v in mat.items():
            mat2[k[0] - min_i, k[1] - min_j] = v

        max_x = max(k[0] for k in mat2)
        max_y = max(k[1] for k in mat2)

        mat = numpy.full((max_y + 1, max_x + 1), ' ')
        for k, v in mat2.items():
            mat[k[1], k[0]] = v
        rows = []
        for i in range(mat.shape[0]):
            rows.append(''.join(mat[i]))
        return "\n".join(rows)


class BiGraph:
    """
    BiGraph representation.

    .. versionadded:: 0.7
    """

    class A:
        "Additional information for a vertex or an edge."

        def __init__(self, kind):
            self.kind = kind

        def __repr__(self):
            return "A(%r)" % self.kind

    class B:
        "Additional information for a vertex or an edge."

        def __init__(self, name, content, onnx_name):
            if not isinstance(content, str):
                raise TypeError(  # pragma: no cover
                    "content must be str not %r." % type(content))
            self.name = name
            self.content = content
            self.onnx_name = onnx_name

        def __repr__(self):
            return "B(%r, %r, %r)" % (self.name, self.content, self.onnx_name)

    def __init__(self, v0, v1, edges):
        """
        :param v0: first set of vertices (dictionary)
        :param v1: second set of vertices (dictionary)
        :param edges: edges
        """
        if not isinstance(v0, dict):
            raise TypeError("v0 must be a dictionary.")
        if not isinstance(v1, dict):
            raise TypeError("v0 must be a dictionary.")
        if not isinstance(edges, dict):
            raise TypeError("edges must be a dictionary.")
        self.v0 = v0
        self.v1 = v1
        self.edges = edges
        common = set(self.v0).intersection(set(self.v1))
        if len(common) > 0:
            raise ValueError(
                "Sets v1 and v2 have common nodes (forbidden): %r." % common)
        for a, b in edges:
            if a in v0 and b in v1:
                continue
            if a in v1 and b in v0:
                continue
            if b in v1:
                # One operator is missing one input.
                # We add one.
                self.v0[a] = BiGraph.A('ERROR')
                continue
            raise ValueError(
                "Edges (%r, %r) not found among the vertices." % (a, b))

    def __str__(self):
        """
        usual
        """
        return "%s(%d v., %d v., %d edges)" % (
            self.__class__.__name__, len(self.v0),
            len(self.v1), len(self.edges))

    def __iter__(self):
        """
        Iterates over all vertices and edges.
        It produces 3-uples:

        * 0, name, A: vertices in *v0*
        * 1, name, A: vertices in *v1*
        * -1, name, A: edges
        """
        for k, v in self.v0.items():
            yield 0, k, v
        for k, v in self.v1.items():
            yield 1, k, v
        for k, v in self.edges.items():
            yield -1, k, v

    def __getitem__(self, key):
        """
        Returns a vertex is key is a string or an edge
        if it is a tuple.

        :param key: vertex or edge name
        :return: value
        """
        if isinstance(key, tuple):
            return self.edges[key]
        if key in self.v0:
            return self.v0[key]
        return self.v1[key]

    def order_vertices(self):
        """
        Orders the vertices from the input to the output.

        :return: dictionary `{vertex name: order}`
        """
        order = {}
        for v in self.v0:
            order[v] = 0
        for v in self.v1:
            order[v] = 0
        modif = 1
        n_iter = 0
        while modif > 0:
            modif = 0
            for a, b in self.edges:
                if order[b] <= order[a]:
                    order[b] = order[a] + 1
                    modif += 1
            n_iter += 1
            if n_iter > len(order):
                break
        if modif > 0:
            raise RuntimeError(
                "The graph has a cycle.\n%s" % pprint.pformat(
                    self.edges))
        return order

    def adjacency_matrix(self):
        """
        Builds an adjacency matrix.

        :return: matrix, list of row vertices, list of column vertices
        """
        order = self.order_vertices()
        ord_v0 = [(v, k) for k, v in order.items() if k in self.v0]
        ord_v1 = [(v, k) for k, v in order.items() if k in self.v1]
        ord_v0.sort()
        ord_v1.sort()
        row = [b for a, b in ord_v0]
        col = [b for a, b in ord_v1]
        row_id = {b: i for i, b in enumerate(row)}
        col_id = {b: i for i, b in enumerate(col)}
        matrix = numpy.zeros((len(row), len(col)), dtype=numpy.int32)
        for a, b in self.edges:
            if a in row_id:
                matrix[row_id[a], col_id[b]] = 1
            else:
                matrix[row_id[b], col_id[a]] = 1
        return matrix, row, col

    def display_structure(self, grid=5, distance=5):
        """
        Creates a display structure which contains
        all the necessary steps to display a graph.

        :param grid: align text to this grid
        :param distance: distance to the text
        :return: instance of @see cl AdjacencyGraphDisplay
        """
        def adjust(c, way):
            if way == 1:
                d = grid * ((c + distance * 2 - (grid // 2 + 1)) // grid)
            else:
                d = -grid * ((-c + distance * 2 - (grid // 2 + 1)) // grid)
            return d

        matrix, row, col = self.adjacency_matrix()
        row_id = {b: i for i, b in enumerate(row)}
        col_id = {b: i for i, b in enumerate(col)}

        interval_y_min = numpy.zeros((matrix.shape[0], ), dtype=numpy.int32)
        interval_y_max = numpy.zeros((matrix.shape[0], ), dtype=numpy.int32)
        interval_x_min = numpy.zeros((matrix.shape[1], ), dtype=numpy.int32)
        interval_x_max = numpy.zeros((matrix.shape[1], ), dtype=numpy.int32)
        interval_y_min[:] = max(matrix.shape)
        interval_x_min[:] = max(matrix.shape)

        graph = AdjacencyGraphDisplay()
        for key, value in self.edges.items():
            if key[0] in row_id:
                y = row_id[key[0]]
                x = col_id[key[1]]
            else:
                x = col_id[key[0]]
                y = row_id[key[1]]
            graph.add(x, y, 'cross', label=value.kind, orientation=(1, 0))
            if x < interval_y_min[y]:
                interval_y_min[y] = x
            if x > interval_y_max[y]:
                interval_y_max[y] = x
            if y < interval_x_min[x]:
                interval_x_min[x] = y
            if y > interval_x_max[x]:
                interval_x_max[x] = y

        for k, v in self.v0.items():
            y = row_id[k]
            x = adjust(interval_y_min[y], -1)
            graph.add(x, y, 'text', label=v.kind, orientation=(-1, 0))
            x = adjust(interval_y_max[y], 1)
            graph.add(x, y, 'text', label=k, orientation=(1, 0))

        for k, v in self.v1.items():
            x = col_id[k]
            y = adjust(interval_x_min[x], -1)
            graph.add(x, y, 'text', label=v.kind, orientation=(0, -1))
            y = adjust(interval_x_max[x], 1)
            graph.add(x, y, 'text', label=k, orientation=(0, 1))

        return graph

    def order(self):
        """
        Order nodes. Depth first.
        Returns a sequence of keys of mixed *v1*, *v2*.
        """
        # Creates forwards nodes.
        forwards = {}
        backwards = {}
        for k in self.v0:
            forwards[k] = []
            backwards[k] = []
        for k in self.v1:
            forwards[k] = []
            backwards[k] = []
        modif = True
        while modif:
            modif = False
            for edge in self.edges:
                a, b = edge
                if b not in forwards[a]:
                    forwards[a].append(b)
                    modif = True
                if a not in backwards[b]:
                    backwards[b].append(a)
                    modif = True

        # roots
        roots = [b for b, backs in backwards.items() if len(backs) == 0]
        if len(roots) == 0:
            raise RuntimeError(  # pragma: no cover
                "This graph has cycles. Not allowed.")

        # ordering
        order = {}
        stack = roots
        while len(stack) > 0:
            node = stack.pop()
            order[node] = len(order)
            w = forwards[node]
            if len(w) == 0:
                continue
            last = w.pop()
            stack.append(last)

        return order

    def summarize(self):
        """
        Creates a text summary of the graph.
        """
        order = self.order()
        keys = [(o, k) for k, o in order.items()]
        keys.sort()

        rows = []
        for _, k in keys:
            if k in self.v1:
                rows.append(str(self.v1[k]))
        return "\n".join(rows)

    @staticmethod
    def _onnx2bigraph_basic(model_onnx, recursive=False):
        """
        Implements graph type `'basic'` for function
        @see fn onnx2bigraph.
        """

        if recursive:
            raise NotImplementedError(  # pragma: no cover
                "Option recursive=True is not implemented yet.")
        v0 = {}
        v1 = {}
        edges = {}

        # inputs
        for i, o in enumerate(model_onnx.graph.input):
            v0[o.name] = BiGraph.A('Input-%d' % i)
        for i, o in enumerate(model_onnx.graph.output):
            v0[o.name] = BiGraph.A('Output-%d' % i)
        for o in model_onnx.graph.initializer:
            v0[o.name] = BiGraph.A('Init')
        for n in model_onnx.graph.node:
            nname = n.name if len(n.name) > 0 else "id%d" % id(n)
            v1[nname] = BiGraph.A(n.op_type)
            for i, o in enumerate(n.input):
                c = str(i) if i < 10 else "+"
                nname = n.name if len(n.name) > 0 else "id%d" % id(n)
                edges[o, nname] = BiGraph.A('I%s' % c)
            for i, o in enumerate(n.output):
                c = str(i) if i < 10 else "+"
                if o not in v0:
                    v0[o] = BiGraph.A('inout')
                nname = n.name if len(n.name) > 0 else "id%d" % id(n)
                edges[nname, o] = BiGraph.A('O%s' % c)

        return BiGraph(v0, v1, edges)

    @staticmethod
    def _onnx2bigraph_simplified(model_onnx, recursive=False):
        """
        Implements graph type `'simplified'` for function
        @see fn onnx2bigraph.
        """
        if recursive:
            raise NotImplementedError(  # pragma: no cover
                "Option recursive=True is not implemented yet.")
        v0 = {}
        v1 = {}
        edges = {}

        # inputs
        for o in model_onnx.graph.input:
            v0["I%d" % len(v0)] = BiGraph.B(
                'In', make_hash_bytes(o.type.SerializeToString(), 2), o.name)
        for o in model_onnx.graph.output:
            v0["O%d" % len(v0)] = BiGraph.B(
                'Ou', make_hash_bytes(o.type.SerializeToString(), 2), o.name)
        for o in model_onnx.graph.initializer:
            v0["C%d" % len(v0)] = BiGraph.B(
                'Cs', make_hash_bytes(o.raw_data, 10), o.name)

        names_v0 = {v.onnx_name: k for k, v in v0.items()}

        for n in model_onnx.graph.node:
            key_node = "N%d" % len(v1)
            if len(n.attribute) > 0:
                ats = []
                for at in n.attribute:
                    ats.append(at.SerializeToString())
                ct = make_hash_bytes(b"".join(ats), 10)
            else:
                ct = ""
            v1[key_node] = BiGraph.B(
                n.op_type, ct, n.name)
            for o in n.input:
                key_in = names_v0[o]
                edges[key_in, key_node] = BiGraph.A('I')
            for o in n.output:
                if o not in names_v0:
                    key = "R%d" % len(v0)
                    v0[key] = BiGraph.B('Re', n.op_type, o)
                    names_v0[o] = key
                edges[key_node, key] = BiGraph.A('O')

        return BiGraph(v0, v1, edges)

    @staticmethod
    def onnx_graph_distance(onx1, onx2, verbose=0, fLOG=print):
        """
        Computes a distance between two ONNX graphs. They must not
        be too big otherwise this function might take for ever.
        The function relies on package :epkg:`mlstatpy`.

        :param onx1: first graph (ONNX graph or model file name)
        :param onx2: second graph (ONNX graph or model file name)
        :param verbose: verbosity
        :param fLOG: logging function
        :return: distance and differences

        .. warning::

            This is very experimental and very slow.

        .. versionadded:: 0.7
        """
        from mlstatpy.graph.graph_distance import GraphDistance

        if isinstance(onx1, str):
            onx1 = onnx.load(onx1)
        if isinstance(onx2, str):
            onx2 = onnx.load(onx2)

        def make_hash(init):
            return make_hash_bytes(init.raw_data)

        def build_graph(onx):
            edges = []
            labels = {}
            for node in onx.graph.node:
                if len(node.name) == 0:
                    name = str(id(node))
                else:
                    name = node.name
                for i in node.input:
                    edges.append((i, name))
                for p, i in enumerate(node.output):
                    edges.append((name, i))
                    labels[i] = "%s:%d" % (node.op_type, p)
                labels[name] = node.op_type
            for init in onx.graph.initializer:
                labels[init.name] = make_hash(init)

            g = GraphDistance(edges, vertex_label=labels)
            return g

        g1 = build_graph(onx1)
        g2 = build_graph(onx2)

        dist, gdist = g1.distance_matching_graphs_paths(
            g2, verbose=verbose, fLOG=fLOG, use_min=False)
        return dist, gdist


def onnx2bigraph(model_onnx, recursive=False, graph_type='basic'):
    """
    Converts an ONNX graph into a graph representation,
    edges, vertices.

    :param model_onnx: ONNX graph
    :param recursive: dig into subgraphs too
    :param graph_type: kind of graph it creates
    :return: see @cl BiGraph

    About *graph_type*:

    * `'basic'`: basic graph structure, it returns an instance
        of type @see cl BiGraph. The structure keeps the original
        names.
    * `'simplified'`: simplifed graph structure, names are removed
        as they could be prevent the algorithm to find any matching.

    .. exref::
        :title: Displays an ONNX graph as text

        The function uses an adjacency matrix of the graph.
        Results are displayed by rows, operator by columns.
        Results kinds are shows on the left,
        their names on the right. Operator types are displayed
        on the top, their names on the bottom.

        .. runpython::
            :showcode:

            import numpy
            from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxSub
            from mlprodict.onnx_conv import to_onnx
            from mlprodict.tools import get_opset_number_from_onnx
            from mlprodict.tools.graphs import onnx2bigraph

            idi = numpy.identity(2).astype(numpy.float32)
            opv = get_opset_number_from_onnx()
            A = OnnxAdd('X', idi, op_version=opv)
            B = OnnxSub(A, 'W', output_names=['Y'], op_version=opv)
            onx = B.to_onnx({'X': idi, 'W': idi})
            bigraph = onnx2bigraph(onx)
            graph = bigraph.display_structure()
            text = graph.to_text()
            print(text)

    .. versionadded:: 0.7
    """
    if graph_type == 'basic':
        return BiGraph._onnx2bigraph_basic(
            model_onnx, recursive=recursive)
    if graph_type == 'simplified':
        return BiGraph._onnx2bigraph_simplified(
            model_onnx, recursive=recursive)
    raise ValueError(
        "Unknown value for graph_type=%r." % graph_type)


def onnx_graph_distance(onx1, onx2, verbose=0, fLOG=print):
    """
    Computes a distance between two ONNX graphs. They must not
    be too big otherwise this function might take for ever.
    The function relies on package :epkg:`mlstatpy`.

    :param onx1: first graph (ONNX graph or model file name)
    :param onx2: second graph (ONNX graph or model file name)
    :param verbose: verbosity
    :param fLOG: logging function
    :return: distance and differences

    .. warning::

        This is very experimental and very slow.

    .. versionadded:: 0.7
    """
    return BiGraph.onnx_graph_distance(onx1, onx2, verbose=verbose, fLOG=fLOG)
