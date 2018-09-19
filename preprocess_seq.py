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
df_vle_info = pd.read_csv('processed_data/vle_info.csv')

print('Apply distribution transformations')

transform_list = ["code_module", "code_presentation", "gender", "region",
    "highest_education", "imd_band", "age_band", "disability"]
for i in transform_list:
    df_vle_info[i] = df_vle_info[i].astype('category')

cat_columns = df_vle_info.select_dtypes(['category']).columns
df_vle_info[cat_columns] = df_vle_info[cat_columns].apply(lambda x: x.cat.codes)


#df_vle_info.to_csv('processed_data/test.csv', index=False)

features = []
for f in df_vle_info:
    if f not in ['code_presentation', 'id_student', 'date', 'final_result']:
        features.append(f)
print(features)

print('Making sequences')
X, y_i = nn_util.make_sequences(df_vle_info, features, 'id_student', sequence_len=SEQ_LEN, verbose=True)
pids = df_vle_info.id_student[y_i].values

print('Saving')
np.save('processed_data/seq_X-' + str(SEQ_LEN) + '.npy', X)
np.save('processed_data/seq_yi-' + str(SEQ_LEN) + '.npy', y_i)
np.save('processed_data/seq_pids-' + str(SEQ_LEN) + '.npy', pids)


exit(0)



print('Making session-length full sequences')
X = []
pids = []
labels = []
df_saa = pd.read_csv('processed_data/student_assesment_assessments.csv')
for pid in tqdm(sorted(df_vle_info.id_student.unique())):
    pid_df = df_vle_info[df_vle_info.id_student == pid]
    for sess in sorted(pid_df.date.unique()):
        sess_df = pid_df[pid_df.date == sess]
        X.append(sess_df[features].values)
        pids.append(pid)
        labels.append(df_saa[df_saa.id_student == pid]['score'])

print('Saving')
np.save('processed_data/seq_X-sess.npy', np.array(X))  # Array of arrays since lengths are ragged.
np.save('processed_data/seq_pids-sess.npy', np.array(pids))
np.save('processed_data/seq_y-sess.npy', np.array(labels))
print(labels)
print(pids)
