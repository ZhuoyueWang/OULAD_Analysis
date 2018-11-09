'''
Task:
Create sequences from the action logs to be used for training unsupervised sequence models.
These following variables should be transformed:
    code_module code_presentation gender region highest_education
    imd_band age_band disability

'''
import pandas as pd
import numpy as np
from tqdm import tqdm

import nn_util


SEQ_LEN = 10

print('Loading data')
df_vle_info = pd.read_csv('processed_data/vle_info_0.csv')
count = 0
#print('Apply distribution transformations')
df_vle_info = df_vle_info.groupby(['code_module'])
for index, eachChunk in tqdm(df_vle_info):
    ii = eachChunk.groupby(['id_student'])
    for index1, iiRows in ii:
        features = []
        for f in iiRows:
            if f not in ['code_presentation',  'code_module', 'id_student','final_result']:
                features.append(f)

        print('Making sequences')
        X, y_i = nn_util.make_sequences(iiRows, features, 'id_student', sequence_len=SEQ_LEN, verbose=True)
        pids = eachChunk.id_student[y_i].values

        print('Saving')
        np.save('processed_data/seq/seq_X-' + str(SEQ_LEN) + '_' + str(iiRows['code_module']).split()[1] + '_' + str(iiRows['id_student']).split()[1]+'.npy', X)
        np.save('processed_data/seq/seq_yi-' + str(SEQ_LEN) + '_' + str(iiRows['code_module']).split()[1] + '_' + str(iiRows['id_student']).split()[1]+'.npy', y_i)
        np.save('processed_data/seq/seq_pids-' + str(SEQ_LEN) + '_' + str(iiRows['code_module']).split()[1] + '_' + str(iiRows['id_student']).split()[1]+'.npy', pids)

        print('Making session-length full sequences')
        X = []
        pids = []
        labels = []
        df_saa = pd.read_csv('processed_data/student_assesment_assessments.csv')
        for pid in tqdm(sorted(eachChunk.id_student.unique())):
            pid_df = eachChunk[eachChunk.id_student == pid]
            for sess in sorted(pid_df.date.unique()):
                sess_df = pid_df[pid_df.date == sess]
                X.append(sess_df[features].values)
                pids.append(pid)
                labels.append(df_saa[df_saa.id_student == pid]['score'])

        print('Saving')
        np.save('processed_data/seq/seq_X-sess' + '_' + str(iiRows['code_module']).split()[1] + '_' + str(iiRows['id_student']).split()[1]+ '.npy', np.array(X))  # Array of arrays since lengths are ragged.
        np.save('processed_data/seq/seq_pids-sess' + '_' + str(iiRows['code_module']).split()[1] + '_' + str(iiRows['id_student']).split()[1]+ '.npy', np.array(pids))
        np.save('processed_data/seq/seq_y-sess' + '_' + str(iiRows['code_module']).split()[1] + '_' + str(iiRows['id_student']).split()[1]+ '.npy', np.array(labels))
        count += 1
        #print(labels)
        #print(pids)
