import hashlib
import hmac
import math
import secrets

import numpy as np

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


def file_to_BF(P, gram_dict, BF_len, file_num, dic_size, K, Num_HMAC):
    BFs = []
    for i in range(file_num):
        BF = BF_plain(Num_HMAC,BF_len)  
        for j in range(dic_size):
            if P[i][j] > 0:
                BF.insert(gram_dict[j], K) 
        BFs.append(BF)
    return BFs


def query_to_BF(Qv, BF_len, K, Num_HMAC):
    BF = BF_plain(Num_HMAC, BF_len)  
    for i in range(len(Qv)):
        BF.insert(Qv[i],K) 
    return BF

