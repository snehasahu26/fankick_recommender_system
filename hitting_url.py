# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:05:54 2020

@author: ssahu
"""

import requests
import pandas as pd 

new_interaction= pd.read_csv('new_interaction_data.csv',index_col=0)
li_=new_interaction['_id_x'].unique()[:10]
print(li_)
for i in li_:
    
    resp = requests.get('http://localhost:1024/'+str(i))
    print(resp.text)