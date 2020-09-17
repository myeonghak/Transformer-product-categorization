#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[ ]:


class produce_output():
    
    def __init__(self):
        pass
    
    def convert_to_category(self,data,result_matrix):
        
        argmax = lambda x: x.argmax()
        argmax_applied= list(map(argmax, result_matrix))
        
        array_result=np.array(result_matrix)
        
        max_prob=array_result.max(axis=1)
        
        full_cat=pd.read_csv("full_cat.csv")
        
        result=pd.DataFrame(argmax_applied)
        
        result["max_prob"]=max_prob

#         result["pieces"]=parts
        
        result=result.merge(full_cat,left_on=0,right_on="new_cat",how="left")

        result=result[[0,"_key_x","_key_y","_key"]]
        
        total_result=pd.concat([data,result],axis=1)
        
        total_result=total_result[["PNM",0,"_key_x","_key_y","_key"]]
        
        total_result.columns=["상품명","카테고리넘버","대분류","중분류","소분류"]
        
        total_result=total_result.fillna(" ")
        
        return total_result

