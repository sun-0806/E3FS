import secrets

import numpy as np
import hmac
import hashlib
import math
import random

from ABBT import ABTree, Bmap, ASPE, hash_to_binary, NHMAC
from fuzzy_process import Read_file, Preprocess, Vect_files, gen_e2LSH_family, uniGram_dictionary, \
    uniGram_dictionary_single
from genkey import Keygen
from BF_process import file_to_BF, query_to_BF, HMAC_keys, Vauth_nonleaf, Vauth_leaf

'''
包括密钥生成，搜索验证协议各部分
'''


class Node:
    def __init__(self, Authvalue):
        self._id = None
        self._isleaf = False
        self._Authvalue = Authvalue
        self._H = b''
        self._pi = b''
        self._verid = None
        self.children = None


def indexGen(P, gram_dict, m, file_num, FHlist, dic_size, dictionary, sk, k1, k2, branch, K, Num_Hmac):
    Omap = {}
    Tmap = {}
    BFs, inv_keyword = file_to_BF(P, gram_dict, Num_Hmac, m, file_num, dic_size, dictionary, K)
    Tag_BF = []
    for BF in BFs:
        EBF = ASPE(BF.bit_array, sk, True)
        Tag_BF.append(NHMAC(EBF, k2))
    for i in range(len(dictionary)):
        update_time = 0
        sigma_w = uniGram_dictionary_single(e2LSH_family, dictionary[i], c)
        Omap[str(sigma_w[0])] = update_time
        D_w = inv_keyword[dictionary[i]]
        broot, eroot = ABTree(D_w, len(D_w), BFs, FHlist, m, sk, k1, k2, Tag_BF, branch)
        f_k1 = hmac.new(k1, (str(sigma_w[0]) + str(update_time)).encode(), digestmod=hashlib.sha1).digest()
        Tmap[f_k1] = eroot
        Bmap[f_k1] = eroot._H
    return Omap, Tmap


def GenTrap(Qv, sk, Omap, BF_len, k1, k2, Num_Hmac, K):
    trapdoors = {}
    rq = {}
    update_times = Omap[str(Qv[0])]
    BF = query_to_BF(Qv, Num_Hmac, BF_len, K)
    sco = BF.one
    for a in range(update_times + 1):
        tau = hmac.new(k1, (str(Qv[0]) + str(a)).encode(), digestmod=hashlib.sha1).digest()
        if a == 0:
            trapdoors[a] = []
            trapdoors[a].append(tau)
            ehbf = ASPE(BF.bit_array, sk, False)
            trapdoors[a].append(ehbf)
            sigma_Q, rq[a] = NHMAC(ehbf, k2)
            trapdoors[a].append(sigma_Q)
        else:
            trapdoors[a].append(tau)
            sk_1 = sk
            sk_1[4] = hash_to_binary(tau, BF_len)
            ehbf = ASPE(BF.bit_array, sk_1, False)
            trapdoors[a].append(ehbf)
            sigma_Q, rq[a] = NHMAC(ehbf, k2)
            trapdoors[a].append(sigma_Q)
    return trapdoors, sco, rq, update_times


def search(sco, node, trap):
    R_i, VO_i = [], {}
    root = Tmap[trap[0]]
    search_tree(root, trap[1], trap[2], R_i, VO_i, sco, node)
    return R_i, VO_i


def is_match(BF, trap1, sco):
    sco1 = np.inner(BF, trap1)
    sco1 = np.array(sco1).flatten()[0]
    if math.fabs(sco - sco1) <= 0.1:
        return -1, sco1
    return 0, sco1


def tag_compute(BF, sigma_D, trap1, trap2):
    if isinstance(BF, np.matrix):
        BF = np.array(BF).flatten()
    if isinstance(sigma_D, np.matrix):
        sigma_D = np.array(sigma_D).flatten()
    if isinstance(trap1, np.matrix):
        trap1 = np.array(trap1).flatten()
    if isinstance(trap2[1], np.matrix):
        trap2[1] = np.array(trap2[1]).flatten()
    y1 = np.dot(BF, trap2[1]) + np.dot(sigma_D, trap1)
    y2 = np.dot(sigma_D, trap2[1])
    return y1, y2


def search_tree(root, trap1, trap2, R_i, VO_i, sco, node):
    EBF = root.Authvalue[0]
    sigma_D = root.Authvalue[1]
    flag, y0 = is_match(EBF, trap1, sco)

    if flag == -1:
        if root.isleaf:
            node._id = root.id
            node.children = root.children
            node._Authvalue = root.Authvalue
            node._isleaf = root.isleaf
            node._pi = root.pi
            node._verid = root.verid
            R_i.append(node._id)
            y1, y2 = tag_compute(EBF, sigma_D, trap1, trap2)
            if node._verid not in VO_i:
                VO_i[node._verid] = []
            VO_i[node._verid].append(y0)
            VO_i[node._verid].append(y1)
            VO_i[node._verid].append(y2)
        else:
            node.children = root.children
            node._Authvalue = root.Authvalue
            for j in range(len(root.children)):
                search_tree(root.children[j], trap1, trap2, R_i, VO_i, sco, node.children[j])
    else:
        node._verid = root.verid
        node._H = root.H
        y1, y2 = tag_compute(EBF, sigma_D, trap1, trap2)
        if node._verid not in VO_i:
            VO_i[node._verid] = []
        VO_i[node._verid].append(y0)
        VO_i[node._verid].append(y1)
        VO_i[node._verid].append(y2)


def verify_complete(node):
    if node.children != None:
        for node_i in node.children:
            verify_complete(node_i)
        node._H = Vauth_nonleaf(node)

    else:
        if node._H == b'':
            node._H = Vauth_leaf(node)


def verify_correctness(Ri, Rq, y0, y1, y2):
    alpha = 2
    result = np.dot(Ri, Rq)
    expected = y0 + y1 * alpha + y2 * alpha * alpha

    result = np.array(result).flatten().round()
    expected = np.array(expected).flatten().round()

    if np.array_equal(result, expected):
        return 1
    else:
        return 0


def to_verify_with_blockchain(hash1, hash2):
    if hash2 == hash1:
        return 1
    else:
        return 0


def verify(VO, Rq, node, tau):
    flag_complete = []
    flag_correct = []
    for key1 in VO:
        ri = Bmap[key1]
        Ri = []
        for i in range(2 * m):
            r_ij = int.from_bytes(hmac.new(k2, str(ri[i]).encode(), digestmod=hashlib.sha1).digest(),
                                  byteorder='big') % 10
            Ri.append(r_ij)
        flag1 = verify_correctness(Ri, Rq, VO[key1][0], VO[key1][1], VO[key1][2])
        flag_correct.append(flag1)
    verify_complete(node)
    hash2 = Bmap[tau]

    flag_complete.append(to_verify_with_blockchain(node._H, hash2))
    return flag_correct, flag_complete


for i in range(1):
    file_dir = "E:/medata"
    Flist, FHlist = Read_file(file_dir)
    Flist = Preprocess(Flist)
    dictionary, X_tfidf = Vect_files(Flist, 10)
    P = X_tfidf.toarray()

    file_num, dic_size = P.shape

    m = 1200
    sk = Keygen(m)
    k1 = secrets.token_hex(16).encode()
    k2 = secrets.token_hex(16).encode()

    N_uni = 300
    Num_LSH = 6
    c = 3
    e2LSH_family = gen_e2LSH_family(N_uni, Num_LSH, c)
    gram_set = uniGram_dictionary(e2LSH_family, dictionary, c)

    Num_Hmac = 10
    K = HMAC_keys(Num_Hmac)
    branch = 2

    Omap, Tmap = indexGen(P, gram_set, m, file_num, FHlist, dic_size, dictionary, sk, k1, k2, branch, K, Num_Hmac)

    query_keywords = []
    obj = random.sample(range(10), 2)
    for i in range(1):
        query_keywords.append(dictionary[obj[i]])
    Qv = uniGram_dictionary(e2LSH_family, query_keywords, c)
    traps, sco, rq, update_times = GenTrap(Qv, sk, Omap, m, k1, k2, Num_Hmac, K)
    for key in traps:
        node = Node(None)
        R_i, VO_i = search(sco, node, traps[key])
        print('match_ids=', R_i)
        Rq = []
        for i in range(2 * m):
            q_ij = int.from_bytes(hmac.new(k2, str(rq[key][i]).encode(), digestmod=hashlib.sha1).digest(),
                                  byteorder='big') % 10
            Rq.append(q_ij)
        flag_correct, flag_complete = verify(VO_i, Rq, node, traps[key][0])
        print('flag_correct, flag_complete=', flag_correct, flag_complete)