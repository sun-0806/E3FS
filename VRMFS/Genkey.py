import secrets

import numpy as np


def Keygen(m):
    m1 = generate_invertible_matrix(m)
    m2 = generate_invertible_matrix(m)
    S = np.random.randint(0, 2, [m])
    M1 = np.matrix(m1)
    M2 = np.matrix(m2)

    sk = [np.transpose(M1), np.transpose(M2), np.linalg.inv(M1), np.linalg.inv(M2), S]
    return sk


def Hash_keys(hash_count):
    K = []
    for i in range(hash_count):
        K.append(secrets.token_hex(16).encode())
    return K


def generate_invertible_matrix(size):
    while True:
        matrix = np.random.rand(size, size)

        det = np.linalg.det(matrix)

        if det != 0:
            return matrix
