import math
import random
from TwinBFs import add_twoBFs, auth_leaf, auth_nonleaf
from MulAcc import Acc, process


class Node():
    def __init__(self, value, acc):
        self._value = value
        self._id = None
        self._left = None
        self._right = None
        self._isleaf = False
        self._ciphertext = ''
        self._X = []
        self._acc = acc
        self._hash = b''


def removeElements(L1, L2):
    b = []
    for item in L1:
        if item not in L2:
            b.append(item)
    return b


def balanceTree(nodes, size, BFs, acc_list, keyword_id_list, threshold, hash_count, pk, pairing, K):
    if size == 1:
        pnode = Node(BFs[nodes[0]], acc_list[nodes[0]])
        pnode._isleaf = True
        pnode._X = keyword_id_list[nodes[0]]
        pnode._hash = auth_leaf(pnode)
        pnode._id = nodes[0]
    else:
        leftnodes = random.sample(nodes, math.ceil(size / 2))
        rightnodes = removeElements(nodes, leftnodes)
        size1 = len(leftnodes)
        size2 = len(rightnodes)
        lnode = balanceTree(leftnodes, size1, BFs, acc_list, keyword_id_list, threshold, hash_count, pk, pairing, K)
        rnode = balanceTree(rightnodes, size2, BFs, acc_list, keyword_id_list, threshold, hash_count, pk, pairing, K)
        pbf = add_twoBFs(rnode._value, lnode._value, threshold, hash_count, K)
        X = list(set(rnode._X + lnode._X))
        pacc = Acc(process(X, pairing), pk, pairing)
        pnode = Node(pbf, pacc)
        pnode._left = lnode
        pnode._right = rnode
        pnode._hash = auth_nonleaf(pnode)
        pnode._X = X
    return pnode
