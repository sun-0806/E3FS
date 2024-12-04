import logging
import sys
import time

import numpy as np
import math
import random

from AuthTree import ASPE
from BF_plain import file_to_BF, query_to_BF, HMAC_keys
from fuzzision import Read_file, Preprocess, Vect_files, gen_e2LSH_family, biGram_dictionary
from Genkey import Keygen



def indexGen(P, gram_dict, m, file_num, dic_size, sk, K, Num_HMAC):
    Dmap = {}
    BFs = file_to_BF(P, gram_dict, m, file_num, dic_size, K, Num_HMAC)
    temp = 0
    for i in range(len(BFs)):
        EBF = ASPE(BFs[i].bit_array, sk, True)
        Dmap[i] = EBF
        temp += sys.getsizeof(EBF)
    return Dmap,temp


def GenTrap(Qv, sk, BF_len, K, Num_HMAC): 
    BF = query_to_BF(Qv, BF_len, K, Num_HMAC)
    logger.info(f"query BF:{BF.bit_array}")
    sco = BF.one
    trapdoor = ASPE(BF.bit_array, sk, False)
    return trapdoor, sco


def search(sco, trap, Dmap):
    R_i = []
    for i in range(len(Dmap)):
        flag = is_match(Dmap[i], trap, sco)
        if flag == 1:
            R_i.append(i)
    return R_i


def is_match(BF, trap1, sco):
    sco1 = np.inner(BF, trap1)
    sco1 = np.array(sco1).flatten()[0]
    if math.fabs(sco - sco1) <= 0.1:
        return 1
    return 0


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Start print log")

file_dir = "./medata"
Flist, FHlist = Read_file(file_dir)
Flist = Preprocess(Flist)
dictionary, X_tfidf = Vect_files(Flist, 4000) 
P = X_tfidf.toarray()
file_num, dic_size = P.shape 
m = 1200
sk = Keygen(m)

N_uni = 676
Num_LSH = 6
c = 3  # LSH的分母

Num_Hmac = 10
K = HMAC_keys(Num_Hmac)

e2LSH_family = gen_e2LSH_family(N_uni, Num_LSH, c)
gram_set = biGram_dictionary(e2LSH_family, dictionary, c)


query_keywords = []
obj = random.sample(range(800), 5)
for i in range(5):
    query_keywords.append(dictionary[obj[i]])
Qv = biGram_dictionary(e2LSH_family, query_keywords, c)
traps, sco = GenTrap(Qv, sk, m, K, Num_Hmac)
R_i = search(sco, traps, Dmap)
print('match_ids=', R_i)

