import pandas as pd
path = "C:\Fall2017-HKUST\CSIC5011-Yaoyuan\Final Project\Age\\age\\Age_data.csv"
df = pd.read_csv(path,header = 0)
#for choices, 1 implies that the left is older than the right'
#while -1 implies that right is older than the left
#For completeness, we let left to be the columns with older age'
#In other words, we let the -1 values switch the two columns
#So that the output contains only two columns'
#Next, we may conduct the analysis by applying it to different participants
#To detect their inconsistency
df_choice_left_older = df.loc[lambda x:x.choice ==1,['photo_id_left','photo_id_right']]
df_choice_right_older = df.loc[lambda x:x.choice ==-1,['photo_id_right','photo_id_left']]
df_choice_right_older= df_choice_right_older.rename(columns={"photo_id_right":"photo_id_left","photo_id_left":"photo_id_right"})
df_concat = df_choice_left_older.append(df_choice_right_older)
df_concat.to_csv('age_2column.csv',header=False,index=False)
