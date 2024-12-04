import numpy as np



class BF_plain:

    def __init__(self, m): 
        self._size = m 
        self.bit_array = np.zeros(m, dtype=int)
        self.one = 0

    def insert(self, obj):
        for i in range(10):
            combined_input = f'{i}_{obj}'
            hash_value = int(hashlib.sha1(combined_input.encode()).hexdigest(), 16) % self.size
            if self.bit_array[hash_value] == 0:
                self.one = self.one + 1
                self.bit_array[hash_value] = 1

    @property
    def size(self):
        return self._size


def file_to_BF(P, gram_dict, BF_len, file_num, dic_size):
    BFs = []
    for i in range(file_num):
        BF = BF_plain(BF_len)  
        for j in range(dic_size):
            if P[i][j] > 0:
                BF.insert(gram_dict[j])  
        BFs.append(BF)
    return BFs


def query_to_BF(Qv, BF_len):
    BF = BF_plain(BF_len)  
    for i in range(len(Qv)):
        BF.insert(Qv[i])  
    return BF

