import hmac
import hashlib
import math
import random
from fuzzision import Read_file, Preprocess, Vect_files, gen_e2LSH_family, uniGram_dictionary
from TwinBFs import file_to_BF, HMAC_keys, auth_leaf, auth_nonleaf
from RSA_signture import to_verify_with_public_key, to_sign
from MulAcc import keyGen, Acc, Proved_disjoint, Verify_disjoint, process
from balanceTree import balanceTree

'''
Including key generation, index and trapdoor construction, search and verification protocol parts
'''
class Node():
    def __init__(self, value, acc):
        self._value = value
        self._id = None
        self._left = None
        self._right = None
        self._isleaf = False
        self._X = []
        self._ciphertext = ''
        self._acc = acc
        self._hash = b''


def Gen_file_acc(T_set, pk, pairing):
    '''the accumulator algorithm for verification'''
    acc_list = []
    for Tj in T_set:
        Tj_Zr = process(Tj, pairing)
        acc = Acc(Tj_Zr, pk, pairing)
        acc_list.append(acc)
    return acc_list


def is_match(BF, traps, tags):
    num = 0   # Record the location of the unmatched subtrap
    for trap in traps:
        for twin in trap:
            first = twin[0]
            second = int(hashlib.sha1((twin[1] + BF.r).encode()).hexdigest(), 16) % 2
            if BF._Twins[first][second] == 0:
                return tags[num]
        num = num + 1
    return -1


def search(root, traps, tags, disjoint, node, pk, match_ids):
    '''
    Search for files from the index tree from the top down
    '''
    flag = is_match(root._value, traps, tags)
    if flag == -1:
        if root._isleaf == True:
            node._value = root._value
            node._acc = root._acc
            node._ciphertext = root._ciphertext
            node._left = root._left
            node._right = root._right
            node._id = root._id
            match_ids.append(node._id)
        else:
            node._value = root._value
            node._acc = root._acc
            node._left = root._left
            node._right = root._right
            search(root._left, traps, tags, disjoint, node._left, pk, match_ids)
            search(root._right, traps, tags, disjoint, node._right, pk, match_ids)
    else:
        pi_1, pi_2, coef = Proved_disjoint(process(root._X, pairing), process([flag], pairing), pk, pairing)
        node._acc = root._acc
        node._hash = root._hash
        disjoint.append((flag, pi_1, pi_2, coef, node._acc))


def verify_correctness(node):
    '''
    Verify whether the current node is correct iteratively
    '''
    if node._left != None or node._right != None:
        verify_correctness(node._left)
        verify_correctness(node._right)
        node._hash = auth_nonleaf(node)
    else:
        if node._hash == b'':
            node._hash = auth_leaf(node)


def verify(disjoint, node, g, pk, signature, pairing):
    '''
    the verification method
    '''
    flag_complete = []
    for temp in disjoint:
        flag1 = Verify_disjoint(temp[4], Acc(process([temp[0]], pairing), pk, pairing), temp[1], temp[2], temp[3],
                                pairing, g)
        flag_complete.append(flag1)
    verify_correctness(node)
    flag_correct = to_verify_with_public_key(signature, node._hash)
    return flag_correct, flag_complete


def indexGen(P, S_set, Num_HMAC, threshold, K, file_num, dic_size, pk, pairing):
    '''
    the index generation method
    '''
    BFs, T_set = file_to_BF(P, S_set, Num_HMAC, threshold, K, file_num, dic_size)
    acc_list = Gen_file_acc(T_set, pk, pairing)

    Nodes = list(range(file_num))
    root = balanceTree(Nodes, file_num, BFs, acc_list, T_set, threshold, Num_HMAC, pk, pairing, K)
    signture = to_sign(root._hash)
    return root, signture


def GenTrap(Qv, K, Num_HMAC, threshold):  
    '''
    the trapdoor generation method
    '''
    traps = []
    tags = []
    for v in Qv:
        trap = []
        tag = ''
        for i in range(Num_HMAC):
            a = hmac.new(K[i], v.encode(), digestmod=hashlib.sha1).hexdigest()
            first = i * math.ceil(threshold) + int(a, 16) % math.ceil(threshold)
            b = hmac.new(K[-1], str(first).encode(), digestmod=hashlib.sha1).hexdigest()
            trap.append([first, b])
            tag = tag + a
        traps.append(trap)
        tags.append(int(hashlib.sha1(tag.encode()).hexdigest(), 16))
    return traps, tags


file_dir = "./medata"
Flist = Read_file(file_dir)
Flist = Preprocess(Flist)
dictionary, P = Vect_files(Flist, 4000)
file_num, dic_size = P.shape

g, pk, pairing = keyGen(dic_size)

N_uni = 160
Num_LSH = 8
c = 3  
e2LSH_family = gen_e2LSH_family(N_uni, Num_LSH, c)
S_set = uniGram_dictionary(e2LSH_family, dictionary, c)

Num_HMAC = 10  
K = HMAC_keys(Num_HMAC)
threshold = 120  

root, signture = indexGen(P, S_set, Num_HMAC, threshold, K, file_num, dic_size, pk, pairing)

query_keywords = []
object = random.sample(range(200), 2)
for i in range(2):
    query_keywords.append(dictionary[object[i]])
Qv = uniGram_dictionary(e2LSH_family, query_keywords, c)
traps, tags = GenTrap(Qv, K, Num_HMAC, threshold)

disjoint, match_ids = [], []
node = Node(None, None)
search(root, traps, tags, disjoint, node, pk, match_ids)
print('match_ids=', match_ids)
flag_correct, flag_complete = verify(disjoint, node, g, pk, signture, pairing)
print('flag_correct, flag_complete=', flag_correct, flag_complete)
