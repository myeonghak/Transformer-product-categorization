#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

import os 
import sys
import re


# In[5]:


import sentencepiece as spm
from keras.preprocessing import text, sequence


# In[ ]:


def 구간(ascii_value):
    if 48<=ascii_value<=57:
        return 0
    elif 65<=ascii_value<=122:
        return 1
    else:
        return 2


# In[3]:


class Preprocessor():
    def __init__(self):
        
        self.SPM_orig = spm.SentencePieceProcessor()
        self.SPM_orig.load("sentencepiece_trsfm.model")
    
    def remove_punct(self,x):
        pattern = re.compile('[^a-zA-Z0-9가-힣]')
        
        x = str(x).strip()
        x = re.sub(pattern, ' ', x)
        return x
    
    def sep_words(self, text):
        
        save_idx=[]
        for i in range(1,len(text)):
            if 구간(ord(text[i]))==구간(ord(text[i-1])):
                pass
            else:
                save_idx.append(i-1)

        count=0

        for k in range(len(save_idx)):
            count+=1
            text=text[:save_idx[k]+count]+" "+text[save_idx[k]+count:]

        return text
    
    
    
    def remove_multispace(self, x):
        x = str(x).strip()
        x = re.sub(' +', ' ',x)

        return x
    
    
    def spm_encoding(self,x):
        return self.SPM_orig.encode_as_ids(x)

    def spm_token(self, x):
        return self.SPM_orig.encode_as_pieces(x)
    
    def padding(self,data,token_col, max_len = 75):
        padded_tokens=sequence.pad_sequences(data[token_col], max_len, padding="post")
    
        return padded_tokens
    
    

