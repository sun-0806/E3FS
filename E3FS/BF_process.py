import hashlib
import hmac
import math
import secrets
import sys

import numpy as np


class Node():
    def __init__(self, value, acc):
        self._value = value
        self._left = None
        self._right = None
        self._isleaf = False
        self._id = None
        self._X = []
        self._ciphertext = ''
        self._acc = acc
        self._hash = b''


def HMAC_keys(hash_count):
    K = []
    for i in range(hash_count):
        K.append(secrets.token_hex(16).encode())
    return K


class BF_plain:

    def __init__(self, hash_count, m):
        self._size = m
        self._hash_count = hash_count
        self.bit_array = np.zeros(m, dtype=int)
        self.one = 0

    def insert(self, obj, K):
        for i in range(self._hash_count):
            a = hmac.new(K[i], obj.encode(), digestmod=hashlib.sha1).hexdigest()
            hash_value = int(a, 16) % math.ceil(self.size)
            if self.bit_array[hash_value] == 0:
                self.one = self.one + 1
                self.bit_array[hash_value] = 1

    @property
    def size(self):
        return self._size


def file_to_BF(P, gram_dict, Num_Hmac, BF_len, file_num, dic_size, dictionary, K):
    BFs = []
    inv_keyword = {keyword: [] for keyword in dictionary}
    for i in range(file_num):
        BF = BF_plain(Num_Hmac, BF_len)
        for j in range(dic_size):
            if P[i][j] > 0:
                BF.insert(gram_dict[j], K)
                keyword = dictionary[j]
                inv_keyword[keyword].append(i)
        BFs.append(BF)
    return BFs, inv_keyword


def query_to_BF(Qv, Num_Hmac, BF_len, K):
    BF = BF_plain(Num_Hmac, BF_len)  #
    for i in range(len(Qv)):
        BF.insert(Qv[i], K)
    return BF


def add_BFs(cnode, m):
    parentBF = [0 for _ in range(m)]
    for j in range(m):
        for i in range(len(cnode)):
            parentBF[j] = parentBF[j] or cnode[i].BFvalue[j]
    return parentBF


def auth_leaf(node):
    a = ''
    a = a + str(node.id) + str(node.pi) + str(node.Authvalue)
    hash = hashlib.sha1(a.encode('utf8')).hexdigest()
    return hash


def auth_nonleaf(node):
    a = '' + str(node.Authvalue)
    for cnode in node.children:
        a = a + str(cnode._H)
    hash = hashlib.sha1(a.encode('utf8')).hexdigest()
    return hash


def Vauth_leaf(node):
    a = ''
    a = a + str(node._id) + str(node._pi) + str(node._Authvalue)
    hash = hashlib.sha1(a.encode('utf8')).hexdigest()
    return hash


def Vauth_nonleaf(node):
    a = '' + str(node._Authvalue)
    for cnode in node.children:
        a = a + str(cnode._H)
    hash = hashlib.sha1(a.encode('utf8')).hexdigest()
    return hash
