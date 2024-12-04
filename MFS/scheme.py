import logging
import time

import numpy as np
import math
import random

from AuthTree import ASPE
from BF_plain import file_to_BF, query_to_BF
from fuzzision import Read_file, Preprocess, Vect_files, gen_e2LSH_family, biGram_dictionary
from Genkey import Keygen1


'''
包括密钥生成，搜索验证协议各部分
'''


def indexGen(P, gram_dict, m, file_num, dic_size, sk):
    '''
    索引创建
    '''
    Dmap = {}
    BFs = file_to_BF(P, gram_dict, m, file_num, dic_size)
    for i in range(len(BFs)):
        EBF = ASPE(BFs[i].bit_array, sk, True)
        Dmap[i]= EBF
    return Dmap


def GenTrap(Qv, sk, BF_len):  # tag:从gram_dict中随机选取若干个
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
    """
    判断是否满足匹配条件，返回一个flag和相似度得分
    """
    sco1 = np.inner(BF, trap1)
    sco1 = np.array(sco1).flatten()[0]
    logger.info(f"预期sco:{sco}, 实际sco:{sco1}")
    if math.fabs(sco - sco1) <= 0.1:
        return 1
    return 0



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Start print log")

file_dir = "E:/data/medtest1/medtest/medtest500"
Flist, FHlist = Read_file(file_dir)
Flist = Preprocess(Flist)
dictionary, P = Vect_files(Flist, 800)  # 200是提取关键词总数
file_num, dic_size = P.shape  # dic_size是所有文件关键词提取个数
logger.info("file processing over")
m = 1200
sk = Keygen1(m)

N_uni = 676
Num_LSH = 8
c = 3  # LSH的分母
e2LSH_family = gen_e2LSH_family(N_uni, Num_LSH, c)
gram_set = biGram_dictionary(e2LSH_family, dictionary, c)
logger.info("indexGen start")
time_start1 = time.time() #开始计时
Dmap = indexGen(P, gram_set, m, file_num, dic_size, sk)
time_end1 = time.time()    #结束计时
logger.info(f"indexGen over:{time_end1-time_start1}")
query_keywords = []
obj = random.sample(range(800), 5)
for i in range(5):
    query_keywords.append(dictionary[obj[i]])
Qv = biGram_dictionary(e2LSH_family, query_keywords, c)
logger.info("GenTrap start")
time_start2 = time.time() #开始计时
traps, sco = GenTrap(Qv, sk, m)
time_end2 = time.time()    #结束计时
logger.info(f"GenTrap over:{time_end2-time_start2}")

logger.info("search start")
time_start3 = time.time()  # 开始计时
R_i= search(sco, traps, Dmap)
time_end3 = time.time()  # 结束计时
logger.info(f"search over:{time_end3 - time_start3}")
print('match_ids=', R_i)
