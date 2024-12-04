import random
import numpy as np


def ASPE(BF, sk, isIndex):
    a = len(BF)
    I_1, I_2 = np.zeros(a), np.zeros(a)
    Q_1, Q_2 = np.zeros(a), np.zeros(a)
    EI_1, EI_2 = np.zeros(a), np.zeros(a)
    QEI_1, QEI_2 = np.zeros(a), np.zeros(a)
    r = random.randint(1, 5)
    for i in range(len(sk[2])):
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
        EI_1 = np.inner(sk[0].reshape((a, a)), I_1)
        EI_2 = np.inner(sk[1].reshape((a, a)), I_2)
        EI = np.concatenate([EI_1, EI_2]).reshape(1, -1)
        return EI
    else:
        QEI_1 = np.inner(sk[2].reshape((a, a)), Q_1)
        QEI_2 = np.inner(sk[3].reshape((a, a)), Q_2)
        QEI = np.concatenate([QEI_1, QEI_2]).reshape(1, -1)
        return QEI
