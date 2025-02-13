import hashlib
import hmac
import math
import secrets
import numpy as np


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


class BF_plain:
     '''
    Definition of Bloom Filter.
    '''

    def __init__(self, hash_count, m):  
        '''
        Initialize Bloom filter
        :param hash_count: the number of hash functions
        :param m: The length of Bloom Filter
        '''
        self._size = m
        self._hash_count = hash_count
        self.bit_array = np.zeros(m, dtype=int)
        self.one = 0

    def insert(self, obj, K):
        '''
        The insertion of Bloom filter
        :param obj: Insert Object
        :param K: PRF keys
        '''
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
    '''
    Process the file set, generate the Bloom filter index of the files.
    :param P: TF-IDF matrix representation of file list
    :param gram_dict: the unigram dictionary of keywords
    :param Num_Hmac: the number of hash functions
    :param BF_len: the length of Bloom Filter
    :param file_num: the number of files
    :param dic_size: the number of keywords
    :param K: PRF keys
    :return: the Bloom filter list of the file set
    '''
    BFs = []
    for i in range(file_num):
        BF = BF_plain(Num_HMAC,BF_len) 
        for j in range(dic_size):
            if P[i][j] > 0:
                BF.insert(gram_dict[j], K)  
        BFs.append(BF)
    return BFs


def query_to_BF(Qv, BF_len, K, Num_HMAC):
    '''
    Process the query, generate the Bloom filter for the query.
    :param Qv: the unigram dictionary of the query
    :param Num_Hmac: the number of hash functions
    :param BF_len: the length of Bloom Filter
    :param K: PRF keys
    :return: the Bloom filter of the query
    '''
    BF = BF_plain(Num_HMAC, BF_len)  
    for i in range(len(Qv)):
        BF.insert(Qv[i],K)  
    return BF


