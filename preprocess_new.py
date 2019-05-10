import pandas as pd
from tqdm import tqdm
import numpy as np


'''
Get merged data and do preprocess
'''
student_info_file = "../anonymisedData/studentInfo.csv"
student_vle_file = "../anonymisedData/studentVle.csv"
vle_file = "../anonymisedData/vle.csv"

df_student_info = pd.read_csv(student_info_file, sep=',', engine='python', header=0)
df_student_info = df_student_info.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

transform_list = ['code_module','code_presentation', 'gender','region',
    'highest_education', 'imd_band', 'age_band', 'disability', 'final_result']
for i in transform_list:
    df_dummy = pd.get_dummies(df_student_info[i])
    df_student_info = pd.concat([df_student_info, df_dummy], axis=1)
df_student_info = df_student_info.drop(columns=transform_list)

df_student_vle = pd.read_csv(student_vle_file, sep=',', engine='python', header=0,nrows=2500000)
df_student_vle = df_student_vle.drop(['code_module', 'code_presentation'], axis=1)
df_student_vle = df_student_vle.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
df_student_vle = df_student_vle.groupby('id_student').filter(lambda x : len(x)>10)

df_vle = pd.read_csv(vle_file, sep=',', engine='python', header=0)
df_vle = df_vle.drop(['code_module', 'code_presentation', 'week_from', 'week_to'], axis=1)

temp = df_vle.merge(df_student_vle, on="id_site", how="outer").dropna()
temp = temp.drop(['id_site'], axis=1)
cols = temp.columns.tolist()
cols = cols[1:]+ cols[:1]
temp = temp[cols]
transform_list = ['activity_type']
for i in transform_list:
    df_dummy = pd.get_dummies(temp[i])
    temp = pd.concat([temp, df_dummy], axis=1)
temp = temp.drop(columns=transform_list)
merged = df_student_info.merge(temp, on="id_student", how="outer").dropna()
merged.to_csv("merged.csv", index=False)
print("ok")
'''
Train-test split
'''
merged = "merged.csv"

df_merged_test = pd.read_csv(merged, sep=',', engine='python', header=0)
df_merged_test = df_merged_test.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

gb_test = df_merged_test.groupby('id_student').tail(1)
gb_train = df_merged_test[~df_merged_test.isin(gb_test)].dropna()

gb_test.to_csv("test.csv", index=False)
gb_train.to_csv("train.csv", index=False)
