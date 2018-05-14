#%%
from nltk import sent_tokenize, word_tokenize
import string
import re
import os
from TurkishStemmer import TurkishStemmer
from collections import defaultdict
from math import log

#%%
#'negative'=-1
#'positive'=1
#'notr'=0
data_path="train/"
stopwords_path="stopwords.txt"
#%%
def fill_stopword(stopwords_path):    
    stoplist=[]
    file = open(stopwords_path, 'r') 
    for line in file: 
        token=re.split('\n',line)
        stoplist.append(token[0].strip()) 
    file.close()
    return stoplist

#%% lower for turkisch characters
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
#%%
 
def read_file(file_path,tokens,data_id,data_labels,label,stoplist,stemmer,positional_stem):
    
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
            for each_sentence in tokenized_sents:
                #temp+=[''.join(c for c in s if c not in string.punctuation and not c.isnumeric() ) for s in each_sentence if s not in stoplist]
                temp+=[''.join(c for c in s if c not in string.punctuation and  c.isalpha() ) for s in each_sentence if s not in stoplist]             
                temp = [convert_turkish_char(stemmer.stem(s)) for s in temp ]
                temp = [s for s in temp if len(s)>1 and s not in stoplist]
            
            for i in temp:
                if i not in positional_stem[label]:
                   positional_stem[label][i]=1
                else:
                   positional_stem[label][i]+=1 
                    
                    
            tokens.append(temp)
            data_labels.append(label)
            sentences=fp.readline()
    fp.close()    
    return tokens,data_id,data_labels,positional_stem


#%%

def read_all_file(data_path,stoplist,stemmer):
    data_labels = []
    data_id = []
    tokens = []
    positional_stem = defaultdict()
    positional_stem[0] = {}
    positional_stem[1] = {}
    positional_stem[-1] = {}
    
    for filename in os.listdir(data_path):
        label = 0
        if 'negative' in filename:
            label = -1
        elif 'positive' in filename:
            label = 1
        tokens,data_id,data_labels,positonal_stem = read_file(data_path+filename,tokens,data_id,data_labels,label,stoplist,stemmer,positional_stem)
        
    return tokens,data_id,data_labels,positional_stem    

#%%        
def convert_turkish_char(sentence):
    sentence = re.sub(r'ğ','g',sentence , flags=re.IGNORECASE)            
    sentence = re.sub(r'ç','c',sentence , flags=re.IGNORECASE)            
    sentence = re.sub(r'ş','s',sentence , flags=re.IGNORECASE)            
    sentence = re.sub(r'ü','u',sentence , flags=re.IGNORECASE)            
    sentence = re.sub(r'ö','o',sentence , flags=re.IGNORECASE)            
    return sentence        
         
    
#%%
stemmer = TurkishStemmer()
stoplist=fill_stopword(stopwords_path)     
tokens,data_id,data_labels,positional_stem=read_all_file(data_path,stoplist,stemmer)     

#%%
wordCountsByClass = [dict(),dict(),dict()]#neg,notr,pos  /-1 0 1
for i in range(len(data_labels)):
    for w in tokens[i]:
        if w in wordCountsByClass[data_labels[i]+1]:
            wordCountsByClass[data_labels[i]+1][w]+=1
        else:
            wordCountsByClass[data_labels[i]+1][w]=1

#%%
def p_snt_in_class(snt,cls):
    lenn = sum(wordCountsByClass[cls].values())
    vocLen = len(wordCountsByClass[cls])
    p=log(lenn/(sum(wordCountsByClass[0].values())+sum(wordCountsByClass[1].values())+sum(wordCountsByClass[2].values())))#probability of class
    for w in snt:
        if w in wordCountsByClass[cls]:
            p+=log((wordCountsByClass[cls][w])/(lenn+vocLen))
        else:    
            p+=log(1/(lenn+vocLen))
    return p    
 
#%%
def m_a_p(snt):
    mapcls = 0#default
    p = p_snt_in_class(snt,0)
    for i in range(len(wordCountsByClass)):
        pnew = p_snt_in_class(snt,i)
        if pnew>p:
            p = pnew
            mapcls=i
    return mapcls-1#returns sentiment of max a posteriori class
 
#%%
      