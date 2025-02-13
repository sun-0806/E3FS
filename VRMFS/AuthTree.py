import hmac
import random
import numpy as np
import hashlib


def removeElements(L1, L2):
    b = []
    for item in L1:
        if item not in L2:
            b.append(item)
    return b


def hash_to_binary(input_string, length):
    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
    binary_hash = bin(int(sha256_hash, 16))[2:]
    if len(binary_hash) > length:
        return binary_hash[:length]
    else:
        return binary_hash.zfill(length)





def ASPE(BF, sk, isIndex):
    '''
    ASPE method
    :param BF: Bloom filter
    :param sk: ASPE keys
    :param isIndex: index or query? index:1 query:0
    :return: the encrypted bloom filter
    '''
    a = len(BF)
    I_1, I_2 = np.zeros(a), np.zeros(a)
    Q_1, Q_2 = np.zeros(a), np.zeros(a)
    EI_1, EI_2 = np.zeros(a), np.zeros(a)
    QEI_1, QEI_2 = np.zeros(a), np.zeros(a)
    r = random.randint(1, 5)
    for i in range(a):
        if isIndex:
            if sk[4][i] == 1:
                I_2[i] = I_1[i] = BF[i]
            else:
                I_1[i] = 0.5 * BF[i] + r
                I_2[i] = 0.5 * BF[i] - r
        else:
            if sk[4][i] == 0:
                Q_2[i] = Q_1[i] = BF[i]
            else:
                Q_1[i] = 0.5 * BF[i] + r
                Q_2[i] = 0.5 * BF[i] - r
    if isIndex:
        EI_1 = np.inner(sk[0].reshape(a, a), I_1)
        EI_2 = np.inner(sk[1].reshape(a, a), I_2)
        EI = np.concatenate([EI_1, EI_2]).reshape(1, -1)
        return EI
    else:
        QEI_1 = np.inner(sk[2].reshape(a, a), Q_1)
        QEI_2 = np.inner(sk[3].reshape(a, a), Q_2)
        QEI = np.concatenate([QEI_1, QEI_2]).reshape(1, -1)
        return QEI


def HMAC(EBF, sk_hmac):
     '''
    HMAC method
    :param EBF: the encrypted bloom filter
    :param sk_hmac: HMAC key
    :return: authenticated bloom filter and verifiable tags
    '''
    Tag_0 = EBF
    EBF_copy = np.asarray(EBF).ravel()
    Tag_1 = []
    Tag_r_i = []
    TBF = []
    alpha = 2
    length = EBF.shape[1]
    for i in range(length):
        r_ij = int.from_bytes(hmac.new(sk_hmac, str(EBF_copy[i]).encode(), digestmod=hashlib.sha1).digest(),
                              byteorder='big') % 10
        Tag_1.append((r_ij - EBF_copy[i]) / alpha)
        Tag_r_i.append(r_ij)
    TBF.append(Tag_0)
    TBF.append(Tag_1)
    return TBF, Tag_r_i
