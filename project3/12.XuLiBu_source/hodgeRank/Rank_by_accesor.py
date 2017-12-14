# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 13:38:15 2017

@author: Jason Li
"""

import pandas as pd
path = "C:\Fall2017-HKUST\CSIC5011-Yaoyuan\Final Project\Age\\age\\Age_data.csv"
df = pd.read_csv(path,header = 0)
#We want to know how many annotators participated in the dataset.
pd.Series.unique(df.annotator_id)
#Now we may regenerate the dataframe with annotator_id
df_choice_left_older = df.loc[lambda x:x.choice ==1,['annotator_id','photo_id_left','photo_id_right']]
df_choice_right_older = df.loc[lambda x:x.choice ==-1,['annotator_id','photo_id_right','photo_id_left']]
df_choice_right_older= df_choice_right_older.rename(columns={"photo_id_right":"photo_id_left","photo_id_left":"photo_id_right"})
df_concat = df_choice_left_older.append(df_choice_right_older)
df_concat.to_csv('age_3column.csv',header=False,index=False)
