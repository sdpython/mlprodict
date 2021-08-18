"""
@file
@brief Helpers to speed up the conversion of Lightgbm models or transform it.
"""
import json
import ctypes
from collections import deque


def restore_lgbm_info(tree):
    """
    Restores speed up information to help
    modifying the structure of the tree.
    """

    def walk_through(t):
        if 'tree_structure' in t:
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
    for node in walk_through(tree):
        if 'right_child' in node or 'left_child' in node:
            nodes.append(node)
    return nodes


def dump_booster_model(self, num_iteration=None, start_iteration=0, importance_type='split'):
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
        return self.dump_model()
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
    ret = json.loads(string_buffer.value.decode('utf-8'))
    ret['pandas_categorical'] = json.loads(
        json.dumps(self.pandas_categorical,
                   default=json_default_with_numpy))
    return ret


def dump_lgbm_booster(booster):
    """
    Dumps a Lightgbm booster into JSON.

    :param booster: Lightgbm booster
    :return: json, dictionary with more information
    """
    js = dump_booster_model(booster)
    info = None
    return js, info


def modify_tree_for_rule_in_set(gbm, use_float=False, verbose=0, count=0,  # pylint: disable=R1710
                                info=None):
    """
    LightGBM produces sometimes a tree with a node set
    to use rule ``==`` to a set of values (= in set),
    the values are separated by ``||``.
    This function unfold theses nodes. A child looks
    like the following:

    :param gbm: a tree coming from lightgbm dump
    :param use_float: use float otherwise int first
        then float if it does not work
    :param verbose: verbosity, use :epkg:`tqdm` to show progress
    :param count: number of nodes already changed (origin) before this call
    :param info: addition information to speed up this search
    :return: number of changed nodes (include *count*)

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
        if verbose >= 2:
            from tqdm import tqdm
            loop = tqdm(gbm['tree_info'])
            for i, tree in enumerate(loop):
                loop.set_description("rules tree %d c=%d" % (i, count))
                count = modify_tree_for_rule_in_set(
                    tree, use_float=use_float, count=count,
                    info=None if info is None else info[i])
        else:
            for i, tree in enumerate(gbm['tree_info']):
                count = modify_tree_for_rule_in_set(
                    tree, use_float=use_float, count=count,
                    info=None if info is None else info[i])
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
        count += 1
        if app:
            stack.append(new_node)
    return count
