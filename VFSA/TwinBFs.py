import hmac
import hashlib
import secrets
import math
import numpy as np
import os

'''Generate Bloom filter, build authenticatied index tree, generate trap door'''

class Node():
    '''
    Definition of index tree nodes
    '''
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
        self.T = None


def HMAC_keys(hash_count):
    '''
    PRF Key Generation.
    :param hash_count: the number of hash functions
    :return: PRF keys
    '''
    K = []
    for i in range(hash_count):
        K.append(secrets.token_hex(16).encode())  
    return K


class TwinBF():
    '''
    Definition of the data structure Twin Bloom Filter
    '''
    def __init__(self, hash_count, m):  
        self._size = m
        self._bit_size = math.ceil(hash_count * self._size)  
        self._hash_count = hash_count  
        self._Twins = np.zeros((self._bit_size, 2))
        self.r = secrets.token_hex(16)
        self.chosenvec = np.zeros(self._bit_size)

    def initi(self):
        for i in range(self._bit_size):
            chosen = int(hashlib.sha1((str(i + int(self.r, 16))).encode()).hexdigest(), 16) % 2
            self._Twins[i][chosen] = 0
            self._Twins[i][1 - chosen] = 1

    def insert(self, Sj, K):
        tj = ''
        for i in range(self._hash_count):
            a = hmac.new(K[i], Sj.encode(), digestmod=hashlib.sha1).hexdigest()
            first = i * math.ceil(self._size) + int(a, 16) % math.ceil(self._size)
            b = hmac.new(K[-1], str(first).encode(), digestmod=hashlib.sha1).hexdigest()
            second = int(hashlib.sha1((b + self.r).encode()).hexdigest(), 16) % 2
            self.chosenvec[first] = 1
            self._Twins[first][second] = 1
            self._Twins[first][1 - second] = 0
            tj = tj + a
        return int(hashlib.sha1(tj.encode()).hexdigest(), 16)


def file_to_BF(P, S_set, Num_HMAC, threshold, K, file_num, dic_size):
    '''
    Process the file set, generate the Twin Bloom filter index of the files.
    '''
    BFs = []
    T_set = []
    for i in range(file_num):
        BF = TwinBF(Num_HMAC, threshold)  
        BF.initi()  
        Ti = []
        for j in range(dic_size):
            if P[i][j] > 0:
                tj = BF.insert(S_set[j], K)  
                Ti.append(tj)
        BFs.append(BF)
        T_set.append(Ti)
    return BFs, T_set


def add_twoBFs(BF1, BF2, m, Num_HMAC, K):
    '''
    Merge two nodes and return the twin Bloom filter after merging nodes
    '''
    parentBF = TwinBF(Num_HMAC, m)  
    for i in range(parentBF._bit_size):
        b = hmac.new(K[-1], str(i).encode(), digestmod=hashlib.sha1).hexdigest()
        chosen = int(hashlib.sha1((b + parentBF.r).encode()).hexdigest(), 16) % 2
        parentBF._Twins[i][chosen] = BF1.chosenvec[i] or BF2.chosenvec[i]
        parentBF.chosenvec[i] = parentBF._Twins[i][chosen]
        parentBF._Twins[i][1 - chosen] = 1 - parentBF._Twins[i][chosen]
    return parentBF


def auth_leaf(node):
    '''
    Authentication leaf node and return the hash of authentication leaf node
    '''
    a = ''
    for item in node._value._Twins:
        a = a + str(item[0]) + str(item[1])
    a = a + node._ciphertext
    a = a + str(node._acc)
    node._hash = hashlib.sha1(a.encode('utf8')).hexdigest()
    return node._hash


def auth_nonleaf(node):
     '''
    Authentication nonleaf node and return the hash of authentication nonleaf node
    '''
    a = ''
    for item in node._value._Twins:
        a = a + str(item[0]) + str(item[1])
    a = a + str(node._acc)
    temp = node._left._hash + node._right._hash
    a = a + hashlib.sha1(temp.encode('utf8')).hexdigest()
    node._hash = hashlib.sha1(a.encode('utf8')).hexdigest()
    return node._hash


def Read_file(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):  
        if (len(files) > 0):
            for file in files:
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    file_list.append(file.read())
    return file_list
