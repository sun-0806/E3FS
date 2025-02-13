import hashlib
import hmac
import math
import secrets
import sys

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


def file_to_BF(P, gram_dict, Num_Hmac, BF_len, file_num, dic_size, dictionary, K):
    '''
    Process the file set, generate the Bloom filter index and the inverted index of the files.
    :param P: TF-IDF matrix representation of file list
    :param gram_dict: the unigram dictionary of keywords
    :param Num_Hmac: the number of hash functions
    :param BF_len: the length of Bloom Filter
    :param file_num: the number of files
    :param dic_size: the number of keywords
    :param dictionary: the keyword set
    :param K: PRF keys
    :return: Bloom filter list, inverted index list
    '''
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
    '''
    Process the query, generate the Bloom filter for the query.
    :param Qv: the unigram dictionary of the query
    :param Num_Hmac: the number of hash functions
    :param BF_len: the length of Bloom Filter
    :param K: PRF keys
    :return: the Bloom filter of the query
    '''
    BF = BF_plain(Num_Hmac, BF_len)  #
    for i in range(len(Qv)):
        BF.insert(Qv[i], K)
    return BF


def add_BFs(cnode, m):
    '''
    Merge child nodes
    :param cnode: node set
    :param m: the length of Bloom Filter
    :return: The Bloom filter after merging nodes
    '''
    parentBF = [0 for _ in range(m)]
    for j in range(m):
        for i in range(len(cnode)):
            parentBF[j] = parentBF[j] or cnode[i].BFvalue[j]
    return parentBF


def auth_leaf(node):
    '''
    Authentication leaf node
    :param node: node
    :return: the hash of authentication leaf node
    '''
    a = ''
    a = a + str(node.id) + str(node.pi) + str(node.Authvalue)
    hash = hashlib.sha1(a.encode('utf8')).hexdigest()
    return hash


def auth_nonleaf(node):
    '''
    Authentication nonleaf node
    :param node: node
    :return: the hash of authentication nonleaf node
    '''
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
