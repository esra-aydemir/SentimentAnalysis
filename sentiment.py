from nltk import sent_tokenize, word_tokenize
import string
import re
import os

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
# quotation mark tokenization will be considered later 
def read_file(file_path,tokens,data_id,data_labels,label,stoplist):
    
    with open(file_path, "r",encoding='utf-8') as fp:
        sentences=fp.readline()
    
        while sentences:   
            tokenized_sents = [word_tokenize(sent.lower())  for sent in sent_tokenize(sentences) ]        
            data_id.append(tokenized_sents[0][0])
            temp=[]
            for each_sentence in tokenized_sents:                
                temp+=[''.join(c for c in s if c not in string.punctuation and not c.isnumeric() ) for s in each_sentence if s not in stoplist]
                temp = [s for s in temp if len(s) >1]
            
            tokens.append(temp)
            data_labels.append(label)
            sentences=fp.readline()
    fp.close()    
    return tokens,data_id,data_labels


#%%

def read_all_file(data_path,stoplist):
    data_labels=[]
    data_id=[]
    tokens=[]
    for filename in os.listdir(data_path):
        label=0
        if 'negative' in filename:
            label=-1
        elif 'positive' in filename:
            label=1
        tokens,data_id,data_labels=read_file(data_path+filename,tokens,data_id,data_labels,label,stoplist)
        
    return tokens,data_id,data_labels    

#%%
stoplist=fill_stopword(stopwords_path)     
tokens,data_id,data_labels=read_all_file(data_path,stoplist)     
        
        