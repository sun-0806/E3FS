import hmac
import random

import numpy as np
from BF_process import auth_leaf, auth_nonleaf, add_BFs
import hashlib

Bmap = {}


class Node:
    def __init__(self, BFvalue):
        self._id = None
        self.children = None
        self._isleaf = False
        self._BFvalue = BFvalue
        self._H = b''
        self._pi = b''

    @property
    def BFvalue(self):
        return self._BFvalue


class ENode:
    def __init__(self, Authvalue):
        self._id = None
        self.children = None
        self._isleaf = False
        self._Authvalue = Authvalue
        self._H = b''
        self._pi = b''
        self._verid = None

    @property
    def id(self):
        return self._id

    @property
    def pi(self):
        return self._pi

    @property
    def H(self):
        return self._H

    @property
    def Authvalue(self):
        return self._Authvalue

    @property
    def verid(self):
        return self._verid

    @property
    def isleaf(self):
        return self._isleaf

    @isleaf.setter
    def isleaf(self, value):
        self._isleaf = value


def hash_to_binary(input_string, length):
    sha256_hash = hashlib.sha256(input_string).hexdigest()

    binary_hash = bin(int(sha256_hash, 16))[2:]

    if len(binary_hash) > length:
        return binary_hash[:length]
    else:
        return binary_hash.zfill(length)


def ASPE(BF, sk, isIndex):
    a = len(BF)
    I_1, I_2 = np.zeros(a), np.zeros(a)
    Q_1, Q_2 = np.zeros(a), np.zeros(a)
    r = random.randint(1, 5)
    for i in range(a):
        if isIndex:
            if sk[4][i] == 1:
                I_2[i] = I_1[i] = BF[i]
            else:
                I_1[i] = 0.5 * BF[i] + r
                I_2[i] = 0.5 * BF[i] - r
        else:
            if sk[4][i] == 0:
                Q_2[i] = Q_1[i] = BF[i]
            else:
                Q_1[i] = 0.5 * BF[i] + r
                Q_2[i] = 0.5 * BF[i] - r
    if isIndex:
        EI_1 = np.inner(sk[0].reshape(a, a), I_1)
        EI_2 = np.inner(sk[1].reshape(a, a), I_2)
        EI = np.concatenate([EI_1, EI_2]).reshape(1, -1)
        return EI
    else:
        QEI_1 = np.inner(sk[2].reshape(a, a), Q_1)
        QEI_2 = np.inner(sk[3].reshape(a, a), Q_2)
        QEI = np.concatenate([QEI_1, QEI_2]).reshape(1, -1)
        return QEI


def NHMAC(EBF, sk_hmac):
    Tag_0 = EBF
    EBF_copy = np.asarray(EBF).ravel()
    Tag_1 = []
    Tag_r_i = EBF_copy
    TBF = []
    alpha = 2
    length = EBF.shape[1]
    for i in range(length):
        r_ij = int.from_bytes(hmac.new(sk_hmac, str(EBF_copy[i]).encode(), digestmod=hashlib.sha1).digest(),
                              byteorder='big') % 10
        Tag_1.append((r_ij - EBF_copy[i]) / alpha)
    TBF.append(Tag_0)
    TBF.append(Tag_1)
    return TBF, Tag_r_i


def ABTree(nodes, size, BFs, file_hash, threshold, sk, k1, sk_hmac, Tag_BF, n):
    if size == 0:
        return None
    if size == 1:
        bnode = Node(BFs[nodes[0]].bit_array)
        TBF, Tag_r_i = Tag_BF[nodes[0]]
        ebnode = ENode(TBF)
        ebnode.isleaf = True
        ebnode._id = nodes[0]
        ebnode._pi = file_hash[nodes[0]]
        ebnode._H = auth_leaf(ebnode)
        ebnode._verid = hmac.new(k1, str(ebnode.H).encode(), digestmod=hashlib.sha1).digest()
        Bmap[ebnode.verid] = Tag_r_i

    else:
        split_nodes = [sub_nodes for sub_nodes in np.array_split(nodes, n) if len(sub_nodes) > 0]
        cnode, ecnode = [], []
        temp = len(split_nodes)
        for i in range(temp):
            size = len(split_nodes[i])
            node, enode = ABTree(split_nodes[i], size, BFs, file_hash, threshold, sk, k1, sk_hmac, Tag_BF, n)
            cnode.append(node)
            ecnode.append(enode)
        parent_bf = add_BFs(cnode, threshold)
        bnode = Node(parent_bf)
        ehbf = ASPE(parent_bf, sk, True)
        TBF, Tag_r_i = NHMAC(ehbf, sk_hmac)
        ebnode = ENode(TBF)
        ebnode.children = ecnode
        ebnode._H = auth_nonleaf(ebnode)
        ebnode._verid = hmac.new(k1, str(ebnode.H).encode(), digestmod=hashlib.sha1).digest()
        Bmap[ebnode.verid] = Tag_r_i
    return bnode, ebnode
