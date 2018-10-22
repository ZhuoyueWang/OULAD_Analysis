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
df_vle_info = pd.read_csv('processed_data/vle_info.csv', chunksize=200000)
count = 0
#print('Apply distribution transformations')
for eachChunk in df_vle_info:
'''
    for f in ['studied_credits']:
        eachChunk[f] = np.log(1 + eachChunk[f].values)

    transform_list = ['code_module', 'id_site','code_presentation', 'gender', 'region',
        'highest_education', 'imd_band', 'age_band', 'disability']

    for i in transform_list:
        df_dummy = pd.get_dummies(eachChunk[i])
        eachChunk = pd.concat([eachChunk, df_dummy], axis=1)
    eachChunk = eachChunk.drop(columns=transform_list)
'''
    features = []
    for f in eachChunk:
        if f not in ['code_presentation',  'id_student','final_result']:
            features.append(f)

    print('Making sequences')
    X, y_i = nn_util.make_sequences(eachChunk, features, 'id_student', sequence_len=SEQ_LEN, verbose=True)
    pids = eachChunk.id_student[y_i].values

    #eachChunk.to_csv('processed_data/test.csv', index=False)
    print('Saving')
    np.save('processed_data/seq_X-' + str(SEQ_LEN) + '_' + str(count) + '.npy', X)
    np.save('processed_data/seq_yi-' + str(SEQ_LEN) + '_' + str(count) + '.npy', y_i)
    np.save('processed_data/seq_pids-' + str(SEQ_LEN) + '_' + str(count) + '.npy', pids)

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
    np.save('processed_data/seq_X-sess' + '_' + str(count) + '.npy', np.array(X))  # Array of arrays since lengths are ragged.
    np.save('processed_data/seq_pids-sess' + '_' + str(count) + '.npy', np.array(pids))
    np.save('processed_data/seq_y-sess' + '_' + str(count) + '.npy', np.array(labels))
    count += 1
    #print(labels)
    #print(pids)
