import pandas as pd
from tqdm import tqdm
import numpy as np
import nn_util

train_file = "train.csv"
test_file = "test.csv"

df_train = pd.read_csv(train_file, sep=',', engine='python', header=0)

features = []
for f in df_train:
    features.append(f)
print(features)

print('Making sequences')
X, y_i = nn_util.make_sequences(df_train, features, 'id_student', sequence_len=10, verbose=True)
pids = df_train.id_student[y_i].values
print(pids)

print('Saving')
np.save('seq_X-' + str(10) + '.npy', X)
np.save('seq_yi-' + str(10) + '.npy', y_i)
np.save('seq_pids-' + str(10) + '.npy', pids)
print("X\n")
print(X[0])
print("y_i\n")
print(y_i[0])
print("pid\n")
print(pids[0])
