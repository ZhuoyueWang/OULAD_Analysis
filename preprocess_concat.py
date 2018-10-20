'''
Task:
1. concatenate student_info and student_vle files
    expected file header contains:
    "code_module", "code_presentation", "id_student", "id_site", "date",
        "sum_click", "gender", "region", "highest_education", "imd_band",
        "age_band", "num_of_prev_attempts", "studied_credits", "disability", "final_result"
2. concatenate student_assessment and assessments files
    expected file header contains:
    "id_assessment", "id_student", "date_submitted", "is_banked", "score",
        "code_module", "code_presentation", "assessment_type", "date", "weight"

studentVle: 10655281 lines
'''

import os
import datetime
import time
import pandas as pd
from tqdm import tqdm




student_info_file = "../anonymisedData/studentInfo.csv"
student_vle_file = "../anonymisedData/studentVle.csv"

print("Start to load files for creating vle_info.csv")

dfs = []

df_student_info = pd.read_csv(student_info_file, sep=',', engine='python', header=0)
df_student_vle = pd.read_csv(student_vle_file, sep=',', engine='python', header=0, chunksize=200000)

for j in df_student_vle:
    headerList = ["code_module", "code_presentation", "id_student", "id_site", "date",
        "sum_click", "gender", "region", "highest_education", "imd_band",
        "age_band", "num_of_prev_attempts", "studied_credits", "disability", "final_result"]
    j = j.groupby(['id_student'])
    for index, rows in tqdm(j):
        rr = rows.groupby(['date'])
        for a, rrRows in rr:
            sum = 0
            getFlag = True
            dict_temp = dict()
            dict_temp["sum_click"] = 0
            for k, row in rrRows.iterrows():
                if getFlag == True:
                    satisfied = df_student_info[df_student_info["id_student"] == row["id_student"]]
                    for idx, i in enumerate(headerList):
                        if idx < 5:
                            dict_temp[i] = row[i]
                        elif idx == 5:
                            dict_temp[i] += int(row[i])
                        else:
                            dict_temp[i] = satisfied[i] #no unique user
                    getFlag = False
                else:
                    dict_temp["sum_click"] += int(row["sum_click"])
            dfs.append(pd.DataFrame(data=dict_temp))
    print('Concatenating files')
    df = pd.concat(dfs, ignore_index=True)
    print('Save vle_info.csv')
    df.to_csv('processed_data/vle_info.csv', mode='a', index=False)



headerList = ["id_assessment", "id_student", "date_submitted", "is_banked", "score",
    "code_module", "code_presentation", "assessment_type", "date", "weight"]

student_assessment_file = "../anonymisedData/studentAssessment.csv"
assessments_file = "../anonymisedData/assessments.csv"

print("Start to load files for creating student_assesment_assessments.csv")

dfs = []

df_student_assessment_file = pd.read_csv(student_assessment_file, sep=',', engine='python', header=0, nrows = 1000)
df_assessments_file = pd.read_csv(assessments_file, sep=',', engine='python', header=0)

print("Successfully loaded")


for index, row in tqdm(df_student_assessment_file.iterrows()):
    dict_temp = dict()
    satisfied = df_assessments_file[df_assessments_file[headerList[0]] == row[headerList[0]]]
    for idx, i in enumerate(headerList):
        if idx <= 4:
            dict_temp[i] = row[i]
        else:
            if satisfied.shape[0] == 1:
                dict_temp[i] = satisfied[i]
            else:
                dict_temp[i] = satisfied[i] #no unique user
    dfs.append(pd.DataFrame(data=dict_temp))

print('Concatenating files')
df = pd.concat(dfs, ignore_index=True)


print('Save student_assesment_assessments.csv')
df.to_csv('processed_data/student_assesment_assessments.csv', index=False)
