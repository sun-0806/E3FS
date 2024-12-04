import logging

import numpy as np
import math
import random

from AuthTree import ASPE, HMAC
from fuzzision import Read_file, Preprocess, Vect_files, gen_e2LSH_family, biGram_dictionary
from Genkey import Keygen
from BF_plain import file_to_BF, query_to_BF, HMAC_keys




def indexGen(P, gram_dict, m, file_num, dic_size, sk, k2,K, Num_HMAC):
    Dmap = {}
    Bmap = {}
    Cmap = {}
    BFs = file_to_BF(P, gram_dict, m, file_num, dic_size, K, Num_HMAC)
    for i in range(len(BFs)):
        EBF = ASPE(BFs[i].bit_array, sk, True)
        Dmap[i] = EBF
        sigma, r_ij = HMAC(EBF, k2)
        Bmap[i] = sigma
        Cmap[i] = r_ij
    return Dmap, Bmap, Cmap


def GenTrap(Qv, sk, BF_len, k2,K,Num_HMAC):
    BF = query_to_BF(Qv, BF_len,K,Num_HMAC)
    sco = BF.one
    trapdoor = ASPE(BF.bit_array, sk, False)
    sigma_Q, rq = HMAC(trapdoor, k2)
    return trapdoor, sco, rq, sigma_Q


def search(sco, trap, Dmap, sigma):
    R_i, VO_i = [], {}
    for i in range(len(Dmap)):
        flag, y0 = is_match(Dmap[i], trap, sco)
        sigma_D = Bmap[i][1]
        if flag == 1:
            R_i.append(i)
        y1, y2 = tag_compute(Dmap[i], sigma_D, trap, sigma)
        VO_i[i] = []
        VO_i[i].append(y0)
        VO_i[i].append(y1)
        VO_i[i].append(y2)
    return R_i, VO_i


def is_match(BF, trap1, sco):
    sco1 = np.inner(BF, trap1)
    sco1 = np.array(sco1).flatten()[0]
    if math.fabs(sco - sco1) <= 0.1:
        return 1, sco1
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


def verify(VO, Rq):
    ri = Cmap[key]
    flag = verify_correctness(ri, Rq, VO[0], VO[1], VO[2])
    return flag


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


file_dir = "./medata"
Flist, FHlist = Read_file(file_dir)
Flist = Preprocess(Flist)
dictionary, P = Vect_files(Flist, 4000)  
file_num, dic_size = P.shape  
m = 1200
sk = Keygen(m)
k2 = secrets.token_hex(16).encode()

N_uni = 676
Num_LSH = 6
c = 3  

Num_Hmac = 10
K = HMAC_keys(Num_Hmac)

e2LSH_family = gen_e2LSH_family(N_uni, Num_LSH, c)
gram_set = biGram_dictionary(e2LSH_family, dictionary, c)
Dmap, Bmap, Cmap = indexGen(P, gram_set, m, file_num, dic_size, sk, k2, K, Num_Hmac) 
query_keywords = []
obj = random.sample(range(1000), 2)
for i in range(2):
    query_keywords.append(dictionary[obj[i]])
Qv = biGram_dictionary(e2LSH_family, query_keywords, c)

traps, sco, rq, sigma = GenTrap(Qv, sk, m, k2, K, Num_Hmac)

R_i, VO_i = search(sco, traps, Dmap, sigma)
print('match_ids=', R_i)
flag_correct = []
for key in VO_i:
    flag = verify(VO_i[key], rq)
    flag_correct.append(flag)
print('flag_correct=', flag_correct)