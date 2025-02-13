import hashlib
import os
import random
import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import logging

table = str.maketrans('', '', string.punctuation)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def Read_file(file_dir):
     '''
    read file
    :param file_dir: Dataset address
    :return: file list and the hash list of the files
    '''
    file_list = []
    file_hash_list = []
    for root, dirs, files in os.walk(file_dir): 
        if len(files) > 0:
            for file in files:
                h = hashlib.new('sha1')
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    file_list.append(file.read())
                    while chunk := file.read(8192):
                        h.update(chunk)
                    file_hash_list.append(h.hexdigest())

    return file_list, file_hash_list


def hash_file(filename, hash_algorithm='sha1'):
    h = hashlib.new(hash_algorithm)
    with open(filename, 'rb') as file:
        while chunk := file.read(8192):
            h.update(chunk)

    return h.hexdigest()


def Preprocess(file_list):
    '''
    Preprocess the file set
    '''
    Flist = []
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for i in range(len(file_list)):
        Words = []
        input_str = (file_list[i])
        input_str = input_str.translate(table)
        input_str = input_str.split()
        for word in input_str:
            word = lemmatizer.lemmatize(word)  
            Words.append(stemmer.stem(word))  
        file = ' '.join(Words)  
        file = re.sub(r"\d+", "", file)  
        Flist.append(file)
    return Flist


def Vect_files(file_list, length):
    '''
    Convert a list of files into TF-IDF vector representation.
    :param file_list: file list
    :param length: maximum number of features (words) to consider in the TF-IDF vector.
    :return: keyword set and TF-IDF matrix representation of file list.
    '''
    tfidf = TfidfVectorizer(max_df=0.4, min_df=0.001, max_features=length, stop_words="english")
    X_tfidf = tfidf.fit_transform(file_list)
    dictionary = tfidf.get_feature_names()
    P = X_tfidf.toarray()
    return dictionary, P


def unigram(word):  
    '''
    Generate unigram vector
    :param word: keyword
    :return: unigram vector
    '''
    vector = np.zeros(160)
    for i in range(len(word)):
        if 'a' <= word[i] <= 'z':
            site = ord(word[i]) - ord('a')
            j = 0
            while vector[site + 26 * j] == 1:
                j = j + 1
            vector[site + 26 * j] = 1
    return vector


def bigram(word):
    '''
    Generate bigram vector
    :param word: keyword
    :return: bigram vector
    '''
    vector = np.zeros(26 * 26)
    for i in range(len(word) - 1):
        site1 = ord(word[i]) - ord('a')
        site2 = ord(word[i + 1]) - ord('a')
        vector[site1 * 26 + site2] = 1
    return vector


def threegram(word):
    '''
    Generate threegram vector
    :param word: keyword
    :return: threegram vector
    '''
    vector = np.zeros(26 * 26 * 26)
    for i in range(len(word) - 2):
        site1 = ord(word[i]) - ord('a')
        site2 = ord(word[i + 1]) - ord('a')
        site3 = ord(word[i + 2]) - ord('a')
        vector[site1 * 26 * 26 + site2 * 26 + site3] = 1
    return vector


def genPara(n, r):  
    '''
    Generate the parameters for p-stable LSH
    :param n: The dimensionality of the input data
    :param r: The bucket width parameter
    :return: the parameters for p-stable LSH
    '''
    a = []
    for i in range(n):
        a.append(random.gauss(0, 1))
    b = random.uniform(0, r)
    return a, b


def gen_e2LSH_family(n, k, r):
     '''
    Generate a family of LSH hash functions.
    :param n: The dimensionality of the input vectors.
    :param k: The number of hash functions in the family
    :param r: The bucket width parameter for LSH
    :return: the LSH family
    '''
    result = []
    for i in range(k):
        result.append(genPara(n, r))
    return result


def gen_HashVals(e2LSH_family, v, r):
    '''
    Compute the LSH hash value for a given vector using a family of E2LSH functions.
    :param e2LSH_family: the LSH family
    :param v: The input vector to be hashed
    :param r: The bucket width parameter.
    :return: The concatenated hash values from all hash functions in the family.
    '''
    hashVals = ''
    for hab in e2LSH_family:
        hashVal = str(int((np.inner(hab[0], v) + hab[1]) // r) + r)
        hashVals = hashVals + hashVal
    return hashVals


def uniGram_dictionary(e2LSH_family, dictionary, r):
    '''
    Generate a unigram dictionary with hash values using E2LSH.
    :param e2LSH_family: the LSH family
    :param dictionary: keyword set
    :param r: The bucket width parameter
    :return: A list containing the hashed representations of each word in the dictionary.
    '''
    gram_dict = []
    for word in dictionary:
        vector = unigram(word)
        hashVals = gen_HashVals(e2LSH_family, vector, r)
        gram_dict.append(hashVals)
    return gram_dict


def uniGram_dictionary_single(e2LSH_family, word, r):
    gram_dict = []
    vector = unigram(word)
    hashVals = gen_HashVals(e2LSH_family, vector, r)
    gram_dict.append(hashVals)
    return gram_dict

def biGram_dictionary(e2LSH_family, dictionary, r):
    gram_dict = []
    for word in dictionary:
        vector = bigram(word)
        hashVals = gen_HashVals(e2LSH_family, vector, r)
        gram_dict.append(hashVals)
    return gram_dict


def threeGram_dictionary(e2LSH_family, dictionary, r):
    gram_dict = []
    for word in dictionary:
        vector = threegram(word)
        hashVals = gen_HashVals(e2LSH_family, vector, r)
        gram_dict.append(hashVals)
    return gram_dict
