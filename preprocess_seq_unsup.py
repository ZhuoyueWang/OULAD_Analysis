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
import glob

import nn_util


SEQ_LEN = 3

print('Loading data')
vle_info_list = list()
for csv_name in glob.glob('processed_data/vle_info_*.csv'):
    vle_info_list.append(pd.read_csv(csv_name, header=0))
vle_info_list = pd.concat(vle_info_list, axis = 0, ignore_index = True)

df_saa = pd.read_csv('processed_data/student_assesment_assessments.csv')
count = 0
#print('Apply distribution transformations')
df_vle_info = vle_info_list.groupby(['code_module'])
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
        print(pids)
        print('Saving')
        labels = []
        labels.append(df_saa[df_saa.id_student == int(pids[0])]['score'])
        fileName = str(iiRows['code_module']).split()[1]

        np.save('processed_data/seq/'+ fileName+ '/seq_X-unsup-' + str(SEQ_LEN) + '_' + str(iiRows['id_student']).split()[1]+'.npy', X)
        np.save('processed_data/seq/'+ fileName+ '/seq_yi-unsup-' + str(SEQ_LEN) + str(iiRows['id_student']).split()[1]+'.npy', y_i)
        np.save('processed_data/seq/'+ fileName+ '/seq_pids-unsup-' + str(SEQ_LEN) + str(iiRows['id_student']).split()[1]+'.npy', pids)
