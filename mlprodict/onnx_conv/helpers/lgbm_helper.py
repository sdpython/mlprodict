"""
@file
@brief Helpers to speed up the conversion of Lightgbm models or transform it.
"""
from collections import deque
import ctypes
import json
import re


def restore_lgbm_info(tree):
    """
    Restores speed up information to help
    modifying the structure of the tree.
    """

    def walk_through(t):
        if 'tree_info' in t:
            yield None
        elif 'tree_structure' in t:
            for w in walk_through(t['tree_structure']):
                yield w
        else:
            yield t
            if 'left_child' in t:
                for w in walk_through(t['left_child']):
                    yield w
            if 'right_child' in t:
                for w in walk_through(t['right_child']):
                    yield w

    nodes = []
    if 'tree_info' in tree:
        for node in walk_through(tree):
            if node is None:
                nodes.append([])
            elif 'right_child' in node or 'left_child' in node:
                nodes[-1].append(node)
    else:
        for node in walk_through(tree):
            if 'right_child' in node or 'left_child' in node:
                nodes.append(node)
    return nodes


def dump_booster_model(self, num_iteration=None, start_iteration=0,
                       importance_type='split', verbose=0):
    """
    Dumps Booster to JSON format.

    Parameters
    ----------
    self: booster
    num_iteration : int or None, optional (default=None)
        Index of the iteration that should be dumped.
        If None, if the best iteration exists, it is dumped; otherwise,
        all iterations are dumped.
        If <= 0, all iterations are dumped.
    start_iteration : int, optional (default=0)
        Start index of the iteration that should be dumped.
    importance_type : string, optional (default="split")
        What type of feature importance should be dumped.
        If "split", result contains numbers of times the feature is used in a model.
        If "gain", result contains total gains of splits which use the feature.
    verbose: dispays progress (usefull for big trees)

    Returns
    -------
    json_repr : dict
        JSON format of Booster.

    .. note::
        This function is inspired from
        the :epkg:`lightgbm` (`dump_model
        <https://lightgbm.readthedocs.io/en/latest/pythonapi/
        lightgbm.Booster.html#lightgbm.Booster.dump_model>`_.
        It creates intermediate structure to speed up the conversion
        into ONNX of such model. The function overwrites the
        `json.load` to fastly extract nodes.
    """
    if getattr(self, 'is_mock', False):
        return self.dump_model(), None
    from lightgbm.basic import (
        _LIB, FEATURE_IMPORTANCE_TYPE_MAPPER, _safe_call,
        json_default_with_numpy)
    if num_iteration is None:
        num_iteration = self.best_iteration
    importance_type_int = FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
    buffer_len = 1 << 20
    tmp_out_len = ctypes.c_int64(0)
    string_buffer = ctypes.create_string_buffer(buffer_len)
    ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
    if verbose >= 2:
        print(  # pragma: no cover
            "[dump_booster_model] call CAPI: LGBM_BoosterDumpModel")
    _safe_call(_LIB.LGBM_BoosterDumpModel(
        self.handle,
        ctypes.c_int(start_iteration),
        ctypes.c_int(num_iteration),
        ctypes.c_int(importance_type_int),
        ctypes.c_int64(buffer_len),
        ctypes.byref(tmp_out_len),
        ptr_string_buffer))
    actual_len = tmp_out_len.value
    # if buffer length is not long enough, reallocate a buffer
    if actual_len > buffer_len:
        string_buffer = ctypes.create_string_buffer(actual_len)
        ptr_string_buffer = ctypes.c_char_p(
            *[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.LGBM_BoosterDumpModel(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            ctypes.c_int(importance_type_int),
            ctypes.c_int64(actual_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))

    WHITESPACE = re.compile(
        r'[ \t\n\r]*', re.VERBOSE | re.MULTILINE | re.DOTALL)

    class Hook(json.JSONDecoder):
        """
        Keep track of the progress, stores a copy of all objects with
        a decision into a different container in order to walk through
        all nodes in a much faster way than going through the architecture.
        """

        def __init__(self, *args, info=None, n_trees=None, verbose=0,
                     **kwargs):
            json.JSONDecoder.__init__(
                self, object_hook=self.hook, *args, **kwargs)
            self.nodes = []
            self.buffer = []
            self.info = info
            self.n_trees = n_trees
            self.verbose = verbose
            self.stored = 0
            if verbose >= 2 and n_trees is not None:
                from tqdm import tqdm  # pragma: no cover
                self.loop = tqdm(total=n_trees)  # pragma: no cover
                self.loop.set_description("dump_booster")  # pragma: no cover
            else:
                self.loop = None

        def decode(self, s, _w=WHITESPACE.match):
            return json.JSONDecoder.decode(self, s, _w=_w)

        def raw_decode(self, s, idx=0):
            return json.JSONDecoder.raw_decode(self, s, idx=idx)

        def hook(self, obj):
            """
            Hook called everytime a JSON object is created.
            Keep track of the progress, stores a copy of all objects with
            a decision into a different container.
            """
            # Every obj goes through this function from the leaves to the root.
            if 'tree_info' in obj:
                self.info['decision_nodes'] = self.nodes
                if self.n_trees is not None and len(self.nodes) != self.n_trees:
                    raise RuntimeError(  # pragma: no cover
                        "Unexpected number of trees %d (expecting %d)." % (
                            len(self.nodes), self.n_trees))
                self.nodes = []
                if self.loop is not None:
                    self.loop.close()
            if 'tree_structure' in obj:
                self.nodes.append(self.buffer)
                if self.loop is not None:
                    self.loop.update(len(self.nodes))
                    if len(self.nodes) % 10 == 0:
                        self.loop.set_description(
                            "dump_booster: %d/%d trees, %d nodes" % (
                                len(self.nodes), self.n_trees, self.stored))
                self.buffer = []
            if "decision_type" in obj:
                self.buffer.append(obj)
                self.stored += 1
            return obj

    if verbose >= 2:
        print("[dump_booster_model] to_json")  # pragma: no cover
    info = {}
    ret = json.loads(string_buffer.value.decode('utf-8'), cls=Hook,
                     info=info, n_trees=self.num_trees(), verbose=verbose)
    ret['pandas_categorical'] = json.loads(
        json.dumps(self.pandas_categorical,
                   default=json_default_with_numpy))
    if verbose >= 2:
        print("[dump_booster_model] end.")  # pragma: no cover
    return ret, info


def dump_lgbm_booster(booster, verbose=0):
    """
    Dumps a Lightgbm booster into JSON.

    :param booster: Lightgbm booster
    :param verbose: verbosity
    :return: json, dictionary with more information
    """
    js, info = dump_booster_model(booster, verbose=verbose)
    return js, info


def modify_tree_for_rule_in_set(gbm, use_float=False, verbose=0, count=0,  # pylint: disable=R1710
                                info=None):
    """
    LightGBM produces sometimes a tree with a node set
    to use rule ``==`` to a set of values (= in set),
    the values are separated by ``||``.
    This function unfold theses nodes.

    :param gbm: a tree coming from lightgbm dump
    :param use_float: use float otherwise int first
        then float if it does not work
    :param verbose: verbosity, use :epkg:`tqdm` to show progress
    :param count: number of nodes already changed (origin) before this call
    :param info: addition information to speed up this search
    :return: number of changed nodes (include *count*)

    A child looks like the following:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import pprint
        from mlprodict.onnx_conv.operator_converters.conv_lightgbm import modify_tree_for_rule_in_set

        tree = {'decision_type': '==',
                'default_left': True,
                'internal_count': 6805,
                'internal_value': 0.117558,
                'left_child': {'leaf_count': 4293,
                               'leaf_index': 18,
                               'leaf_value': 0.003519117642745049},
                'missing_type': 'None',
                'right_child': {'leaf_count': 2512,
                                'leaf_index': 25,
                                'leaf_value': 0.012305307958365394},
                'split_feature': 24,
                'split_gain': 12.233599662780762,
                'split_index': 24,
                'threshold': '10||12||13'}

        modify_tree_for_rule_in_set(tree)

        pprint.pprint(tree)
    """
    if 'tree_info' in gbm:
        if info is not None:
            dec_nodes = info['decision_nodes']
        else:
            dec_nodes = None
        if verbose >= 2:  # pragma: no cover
            from tqdm import tqdm
            loop = tqdm(gbm['tree_info'])
            for i, tree in enumerate(loop):
                loop.set_description("rules tree %d c=%d" % (i, count))
                count = modify_tree_for_rule_in_set(
                    tree, use_float=use_float, count=count,
                    info=None if dec_nodes is None else dec_nodes[i])
        else:
            for i, tree in enumerate(gbm['tree_info']):
                count = modify_tree_for_rule_in_set(
                    tree, use_float=use_float, count=count,
                    info=None if dec_nodes is None else dec_nodes[i])
        return count

    if 'tree_structure' in gbm:
        return modify_tree_for_rule_in_set(
            gbm['tree_structure'], use_float=use_float, count=count,
            info=info)

    if 'decision_type' not in gbm:
        return count

    def str2number(val):
        if use_float:
            return float(val)
        else:
            try:
                return int(val)
            except ValueError:  # pragma: no cover
                return float(val)

    if info is None:

        def recursive_call(this, c):
            if 'left_child' in this:
                c = process_node(this['left_child'], count=c)
            if 'right_child' in this:
                c = process_node(this['right_child'], count=c)
            return c

        def process_node(node, count):
            if 'decision_type' not in node:
                return count
            if node['decision_type'] != '==':
                return recursive_call(node, count)
            th = node['threshold']
            if not isinstance(th, str):
                return recursive_call(node, count)
            pos = th.find('||')
            if pos == -1:
                return recursive_call(node, count)
            th1 = str2number(th[:pos])

            def doit():
                rest = th[pos + 2:]
                if '||' not in rest:
                    rest = str2number(rest)

                node['threshold'] = th1
                new_node = node.copy()
                node['right_child'] = new_node
                new_node['threshold'] = rest

            doit()
            return recursive_call(node, count + 1)

        return process_node(gbm, count)

    # when info is used

    def split_node(node, th, pos):
        th1 = str2number(th[:pos])

        rest = th[pos + 2:]
        if '||' not in rest:
            rest = str2number(rest)
            app = False
        else:
            app = True

        node['threshold'] = th1
        new_node = node.copy()
        node['right_child'] = new_node
        new_node['threshold'] = rest
        return new_node, app

    stack = deque(info)
    while len(stack) > 0:
        node = stack.pop()

        if 'decision_type' not in node:
            continue  # leave

        if node['decision_type'] != '==':
            continue

        th = node['threshold']
        if not isinstance(th, str):
            continue

        pos = th.find('||')
        if pos == -1:
            continue

        new_node, app = split_node(node, th, pos)
        count += 1
        if app:
            stack.append(new_node)

    return count
