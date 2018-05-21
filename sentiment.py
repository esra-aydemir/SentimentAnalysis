#%%
from nltk import sent_tokenize, word_tokenize
import string
import re
import os
from TurkishStemmer import TurkishStemmer
from collections import defaultdict
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from gensim.models import KeyedVectors
import numpy as np
from numpy import linalg as LA
import glob
import random

data_path="../train/"
stopwords_path="stopwords.txt"
#%% preprocess/read file


def fill_stopword(stopwords_path):    
    stoplist=[]
    file = open(stopwords_path, 'r') 
    for line in file: 
        token=re.split('\n',line)
        stoplist.append(token[0].strip()) 
    file.close()
    return stoplist

# lower for turkisch characters
def trlower(str):
    res = ""
    for c in str:
        if c =='ı':
            res=res+'ı'
        elif c =='Ğ'or c =='ğ':
            res=res+'ğ'
        elif c =='Ü' or c=='ü':
            res=res+'ü'
        elif c =='Ş' or c =='ş':
            res=res+'ş'
        elif c =='İ':
            res=res+'i'
        elif c =='Ö' or c =='ö':
            res=res+'ö'
        elif c =='Ç' or c =='ç':
            res=res+'ç'
        else:
            res=res + c.lower()
    return res

def convert_turkish_char(sentence):
    sentence = re.sub(r'ğ','g',sentence , flags=re.IGNORECASE)            
    sentence = re.sub(r'ç','c',sentence , flags=re.IGNORECASE)            
    sentence = re.sub(r'ş','s',sentence , flags=re.IGNORECASE)            
    sentence = re.sub(r'ü','u',sentence , flags=re.IGNORECASE)            
    sentence = re.sub(r'ö','o',sentence , flags=re.IGNORECASE)            
    return sentence        

#use after to lower
def remove_vowels_(string):
    res = ''
    for c in string:
        if c in ['q','w','r','t','y','p','ğ','s','d','f','g','h','j','k','l','ş','z','x','c','v','b','n','m','ç']:
            res+=c
    return res
def read_file(file_path,tokens,data_id,data_labels,label,stoplist,stemmer,positional_stem, all_sentences,distinct_word  ,positional_word , word_vectors,remove_vowels=False):
    
    with open(file_path, "r",encoding='utf-8') as fp:
        sentences=fp.readline()
    
        while sentences:  
            sentences = re.sub('@\S*', ' ', sentences)
            sentences = re.sub('pic\\..*(\s)+', ' ', sentences)
            sentences = re.sub(r"’(\S)*(\s)",' ',sentences)
            sentences = re.sub(r"'(\S)*(\s)",' ',sentences)    
            
            #print(sentences)
            tokenized_sents = [word_tokenize(trlower(sent))  for sent in sent_tokenize(sentences) ]        
            data_id.append(tokenized_sents[0][0])
            temp=[]
            
            count = len(all_sentences)
            for each_sentence in tokenized_sents:
                #temp+=[''.join(c for c in s if c not in string.punctuation and not c.isnumeric() ) for s in each_sentence if s not in stoplist]
                temp+=[''.join(c for c in s if c not in string.punctuation and  c.isalpha() ) for s in each_sentence if s not in stoplist]             
                original_temp = list(filter(None, temp))
                temp = [convert_turkish_char(stemmer.stem(s)) for s in temp ]
                temp = [s for s in temp if len(s)>1 and s not in stoplist]
                if remove_vowels:
                    temp = [remove_vowels_(s) for s in temp]            
            # word in model for word2vec 
            for i in original_temp:
                    distinct_word[i]=0
                    if i in word_vectors:
                        if i not in positional_word:
                            positional_word[i]=[count]
                        else:
                            positional_word[i].append(count)
                
            one_sentence = ' '
            for i in temp:
                one_sentence += ' '+i
                if i not in positional_stem[label]:
                   positional_stem[label][i]=1
                else:
                   positional_stem[label][i]+=1 
                    
                    
            tokens.append(temp)
            data_labels.append(label)
            all_sentences.append(one_sentence)

            sentences=fp.readline()
            
    fp.close()    
    return tokens,data_id,data_labels,positional_stem,all_sentences,distinct_word,  positional_word

def read_all_file(data_path,stoplist,stemmer,word_vectors,remove_vowels=False):
    data_labels = []
    data_id = []
    tokens = []
    all_sentences = []
    positional_stem = defaultdict()
    positional_stem[0] = {}
    positional_stem[1] = {}
    positional_stem[-1] = {}
    distinct_word = defaultdict()
    positional_word = defaultdict()
    
    for filename in os.listdir(data_path):
        label = 0
        if 'negative' in filename:
            label = -1
        elif 'positive' in filename:
            label = 1
        tokens,data_id,data_labels,positonal_stem, all_sentences,distinct_word ,positional_word = read_file(data_path+filename,tokens,data_id,data_labels,label,stoplist,stemmer,positional_stem,all_sentences,distinct_word, positional_word, word_vectors,remove_vowels)
        
    return tokens,data_id,data_labels,positional_stem, all_sentences,distinct_word,positional_word    

#%% read file of tweet3000 dataset:
def read3000tweet():

    def readData(folderPath,label,all_sentences):
        dirlist = glob.glob(folderPath+"/*.txt")
        stoplist=fill_stopword(stopwords_path) 
        for dirr in dirlist:
            with open(dirr, "r",encoding='latin-1') as fp:
                sentences=fp.readline()
                while sentences:  
                    sentences = re.sub('@\S*', ' ', sentences)
                    sentences = re.sub('pic\\..*(\s)+', ' ', sentences)
                    sentences = re.sub(r"’(\S)*(\s)",' ',sentences)
                    sentences = re.sub(r"'(\S)*(\s)",' ',sentences)    
                    
                    #print(sentences)
                    tokenized_sents = [word_tokenize(trlower(sent))  for sent in sent_tokenize(sentences) ]        
                    data_id.append(tokenized_sents[0][0])
                    temp=[]
                    
                    for each_sentence in tokenized_sents:
                        #temp+=[''.join(c for c in s if c not in string.punctuation and not c.isnumeric() ) for s in each_sentence if s not in stoplist]
                        temp+=[''.join(c for c in s if c not in string.punctuation and  c.isalpha() ) for s in each_sentence if s not in stoplist]             
                        temp = [convert_turkish_char(stemmer.stem(s)) for s in temp ]
                        temp = [s for s in temp if len(s)>1 and s not in stoplist]
                        
                    one_sentence = ' '
                    for i in temp:
                        one_sentence += ' '+i
                            
                            
                    tokens.append(temp)
                    data_labels.append(label)
                    all_sentences.append(one_sentence)
        
                    sentences=fp.readline()
            fp.close()  
        return data_labels,tokens, all_sentences
             
    all_sentences=[]
    data_labels=[]
    tokens=[]
    pathToDataset = "3000tweet"
    positives = readData(pathToDataset+"/1",1,all_sentences)
    negatives = readData(pathToDataset+"/-1",-1,all_sentences)  
    neutrals = readData(pathToDataset+"/0",0,all_sentences)
    return all_sentences,data_labels,tokens
    
#%%tf-idf model
# it is not necessary now , but it will be useful for future work    
def get_tfidf(all_sentences):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

    features = tfidf.fit_transform(all_sentences)
    print(features.shape)
    return features

def train_tfidf_model(model, all_sentences,data_labels):
    X_train, X_test, y_train, y_test = train_test_split(all_sentences, data_labels,  test_size=0.30 , random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = model.fit(X_train_tfidf, y_train)   
    y_pred = clf.predict(count_vect.transform(X_test))
    print(metrics.classification_report(y_test, y_pred))
         

    
#%%Word2Vec model
# generate word to vector in dictionary    
def get_dictionary_word2vec(distinct_word, word_vectors):
    dic_word2vec={}
    for i in distinct_word:
        if i in word_vectors:
            dic_word2vec[i]=word_vectors.get_vector(i) 

    return dic_word2vec


# all word vector addition
def get_tweet_vector_addition(num_sentence,dic_word2vec,positional_word):
    sentence_vec = np.zeros([num_sentence,400])
    for each_word in positional_word:
        for sentence_id in positional_word[each_word]:
            sentence_vec[sentence_id]  = np.add(sentence_vec[sentence_id] , dic_word2vec[each_word])
        
    return sentence_vec

# all word vector addition, then take average
def get_tweet_vector_average(num_sentence,dic_word2vec,positional_word):
    sentence_vec = np.zeros([num_sentence,400])
    sentence_count = np.zeros([num_sentence])
    
    for each_word in positional_word:
        for sentence_id in positional_word[each_word]:
            sentence_count[sentence_id] = sentence_count[sentence_id] + 1
            sentence_vec[sentence_id]  = np.add(sentence_vec[sentence_id] , dic_word2vec[each_word])
    
    for i in range(len(sentence_count)):
        if sentence_count[i]==0:
            sentence_count[i] = 1
    
    sentenceVec = [sentence_vec[i]/sentence_count[i]   for i in range(len(sentence_vec))]    
    
    return sentenceVec

def train_word2vec_model(model, all_sentences,data_labels):
    X_train, X_test, y_train, y_test = train_test_split(all_sentences, data_labels,  test_size=0.3 ,random_state = 0)
    clf = model.fit(X_train, y_train)   
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    
#%%
# arrange later , but now it is template    
def main():
    stemmer = TurkishStemmer()
    stoplist=fill_stopword(stopwords_path)     
    tokens,data_id,data_labels,positional_stem,all_sentences, distinct_word , positional_word = read_all_file(data_path,stoplist,stemmer,word_vectors)     
    #dic_word2vec= get_dictionary_word2vec(distinct_word, word_vectors)
    num_sentences = len(all_sentences)
    sentence_vec = get_tweet_vector_addition(num_sentences,dic_word2vec,positional_word)

    #model = KNeighborsClassifier(n_neighbors=5)
    #model = MultinomialNB()    
    model = LogisticRegression(random_state=0)
    #model = LinearSVC()
    #model = NearestCentroid()
    #model.metric='euclidean'
    #train_tfidf_model(model, all_sentences,data_labels)    
    train_word2vec_model(model, sentence_vec,data_labels)  
    
#%% main2:
#'negative'=-1
#'positive'=1
#'notr'=0
#%%preprocess:

stemmer = TurkishStemmer()
stoplist=fill_stopword(stopwords_path)     
tokens,data_id,data_labels,positional_stem,all_sentences, distinct_word , positional_word = read_all_file(data_path,stoplist,stemmer,word_vectors)     

all_sent2,data_label2,tokens2 = read3000tweet()

#%%
X_train, X_test, y_train, y_test = train_test_split(all_sentences, data_labels,  test_size=0.30 , random_state = 0)

c = list(zip(all_sent2,data_label2))
random.shuffle(c)
a, b = zip(*c)
X_train+=list(a)
y_train+=list(b)

def train_tfidf_model_dataset2(model, X_train,X_test,y_train,y_test):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = model.fit(X_train_tfidf, y_train)   
    y_pred = clf.predict(count_vect.transform(X_test))
    print(metrics.classification_report(y_test, y_pred))


         
#%% tf-idf:
model = MultinomialNB()
train_tfidf_model(model,all_sentences,data_labels)
train_tfidf_model_dataset2(model, X_train,X_test,y_train,y_test)

#%% word2vec:
word_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)


num_sentences = len(all_sentences)
dic_word2vec= get_dictionary_word2vec(distinct_word, word_vectors)

sentence_vec_addition = get_tweet_vector_addition(num_sentences,dic_word2vec,positional_word)

sentence_vec_average = get_tweet_vector_average(num_sentences,dic_word2vec,positional_word)

sentence_vec = sentence_vec_addition #= sentence_vec_average

train_word2vec_model(model,all_sentences,data_labels)