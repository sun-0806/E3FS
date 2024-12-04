import time

import numpy as np
import math
import random

from AuthTree import ASPE
from BF_plain import file_to_BF, query_to_BF
from fuzzision import Read_file, Preprocess, Vect_files, gen_e2LSH_family, biGram_dictionary
from Genkey import Keygen


def indexGen(P, gram_dict, m, file_num, dic_size, sk):
    Dmap = {}
    BFs = file_to_BF(P, gram_dict, m, file_num, dic_size)
    for i in range(len(BFs)):
        EBF = ASPE(BFs[i].bit_array, sk, True)
        Dmap[i]= EBF
    return Dmap


def GenTrap(Qv, sk, BF_len):  
    BF = query_to_BF(Qv, BF_len)
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




file_dir = "./medata"
Flist, FHlist = Read_file(file_dir)
Flist = Preprocess(Flist)
dictionary, P = Vect_files(Flist, 4000)  
file_num, dic_size = P.shape  
m = 1200
sk = Keygen(m)

N_uni = 676
Num_LSH = 6
c = 3  
e2LSH_family = gen_e2LSH_family(N_uni, Num_LSH, c)
gram_set = biGram_dictionary(e2LSH_family, dictionary, c)
Dmap = indexGen(P, gram_set, m, file_num, dic_size, sk)
time_end1 = time.time()    
query_keywords = []
obj = random.sample(range(800), 1)
for i in range(1):
    query_keywords.append(dictionary[obj[i]])
Qv = biGram_dictionary(e2LSH_family, query_keywords, c)
traps, sco = GenTrap(Qv, sk, m)
R_i= search(sco, traps, Dmap)
time_end3 = time.time()  
print('match_ids=', R_i)
