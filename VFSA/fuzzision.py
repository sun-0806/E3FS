import os
import random
import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

table =str.maketrans('','', string.punctuation)
def Read_file(file_dir):
    file_list=[]
    for root, dirs, files in os.walk (file_dir):
        if (len(files)>0):
            for file in files:
                path = os.path.join (root,file)
                with open (path, 'r', encoding = 'utf-8', errors = 'ignore') as file:
                    file_list.append (file.read())
    return file_list

def Preprocess(file_list):
    Flist=[]
    lemmatizer=WordNetLemmatizer()
    stemmer=PorterStemmer()
    for i in range(len(file_list)):
        Words=[]
        input_str=(file_list[i])       
        input_str=input_str.translate(table)
        input_str=input_str.split()
        for word in input_str:
            word=lemmatizer.lemmatize(word) 
            Words.append(stemmer.stem(word))
        file=' '.join(Words) 
        file=re.sub(r"\d+","",file)
        Flist.append(file)  
    return Flist

def Vect_files(file_list,length):
    tfidf = TfidfVectorizer(max_df=0.4,min_df=0.001,max_features=length,stop_words = "english")
    X_tfidf = tfidf.fit_transform (file_list)
    dictionary = tfidf.get_feature_names()
    P = X_tfidf.toarray()
    return dictionary,P

def unigram(word):
    vector=np.zeros(160)
    for i in range(len(word)):
        if 'a'<=word[i]<='z':
            site=ord(word[i])-ord('a')
            j=0
            while vector[site+26*j]==1:
                j=j+1
            vector[site+26*j]=1
    return vector
 
def bigram(word):          
    vector=np.zeros(26*26)
    for i in range(len(word)-1):
        site1=ord(word[i])-ord('a')
        site2=ord(word[i+1])-ord('a')
        vector[site1*26+site2]=1
    return vector

def threegram(word):
    vector=np.zeros(26*26*26)
    for i in range(len(word)-2):
        site1=ord(word[i])-ord('a')
        site2=ord(word[i+1])-ord('a')
        site3=ord(word[i+2])-ord('a')
        vector[site1*26*26+site2*26+site3]=1
    return vector
        
def genPara(n, r):
    a = []
    for i in range(n):
        a.append(random.gauss(0, 1))
    b = random.uniform(0, r)
    return a, b

def gen_e2LSH_family(n, k, r):
    result = []
    for i in range(k):
        result.append(genPara(n, r))
    return result

def gen_HashVals(e2LSH_family, v, r):
    hashVals =''
    for hab in e2LSH_family:
        hashVal = str(int((np.inner(hab[0], v) + hab[1]) // r)+r)
        hashVals=hashVals+hashVal
    return hashVals

def uniGram_dictionary(e2LSH_family,dictionary,r):
    gram_dict=[]
    for word in dictionary:
        vector=unigram(word)
        hashVals=gen_HashVals(e2LSH_family,vector,r)
        gram_dict.append(hashVals)
    return gram_dict

def biGram_dictionary(e2LSH_family,dictionary,r):
    gram_dict=[]
    for word in dictionary:
        vector=bigram(word)
        hashVals=gen_HashVals(e2LSH_family,vector,r)
        gram_dict.append(hashVals)
    return gram_dict

def threeGram_dictionary(e2LSH_family,dictionary,r):
    gram_dict=[]
    for word in dictionary:
        vector=threegram(word)
        hashVals=gen_HashVals(e2LSH_family,vector,r)
        gram_dict.append(hashVals)
    return gram_dict






