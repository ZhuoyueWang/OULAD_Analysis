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
    cat_columns = df.select_dtypes([i]).columns
    df_vle_info[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

features = []
for f in df:
    print(f)
    try:
        _ = float(df[f].iloc[0])
        features.append(f)
    except:
        pass
print(features)

print('Making sequences')
X, y_i = nn_util.make_sequences(df, features, 'id_student', sequence_len=SEQ_LEN, verbose=True)
pids = df.student_id[y_i].values

print('Saving')
np.save('processed_data/seq_X-' + str(SEQ_LEN) + '.npy', X)
np.save('processed_data/seq_yi-' + str(SEQ_LEN) + '.npy', y_i)
np.save('processed_data/seq_pids-' + str(SEQ_LEN) + '.npy', pids)


print('Making session-length full sequences')
X = []
pids = []
labels = []
df_saa = df.read_csv('processed_data/student_assesment_assessments.csv')
df_saa['id_student'] = 'student' + df_saa['Student Id'].astype(str)
for pid in tqdm(sorted(df.id_student.unique())):
    pid_df = df[df.id_student == pid]
    for sess in sorted(pid_df.session_num.unique()):
        sess_df = pid_df[pid_df.session_num == sess]
        X.append(sess_df[features].values)
        pids.append(pid)
        labels.append(np.nan if sess == 'session1' else
                      df_saa[df_saa.id_student == pid]['Session ' + sess[-1]].iloc[0])

print('Saving')
np.save('processed_data/seq_X-sess.npy', np.array(X))  # Array of arrays since lengths are ragged.
np.save('processed_data/seq_pids-sess.npy', np.array(pids))
np.save('processed_data/seq_y-sess.npy', np.array(labels))
print(labels)
print(pids)
