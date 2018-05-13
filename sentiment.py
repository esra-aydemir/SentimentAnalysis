from nltk import sent_tokenize, word_tokenize
import string
import re
import os
from TurkishStemmer import TurkishStemmer
from collections import defaultdict

#%%
#'negative'=-1
#'positive'=1
#'notr'=0
data_path="/home/hilal/Documents/cmpe493/term-project/train/"
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
            tokenized_sents = [word_tokenize(sent.lower())  for sent in sent_tokenize(sentences) ]        
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
        