import numpy as np
import pandas as pd
from keras import models, layers, optimizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

import nn_util

train_file = "train.csv"
test_file = "test.csv"

df_train = pd.read_csv(train_file, sep=',', engine='python', header=0)
df_test = pd.read_csv(test_file, sep=',', engine='python', header=0)

nn_util.limit_keras_gpu_mem(.1)
np.random.seed(11798)  # So that data splitting is the same each run.

print('Loading unsupervised data')
X = np.load('seq_X-10.npy')
pids = np.load('seq_pids-10.npy')

ae_in = layers.Input([None, X[0].shape[-1]])
m = layers.Conv1D(50, kernel_size=5, padding='valid')(ae_in)
m = layers.advanced_activations.ELU(.3)(m)
m = layers.Conv1D(50, kernel_size=2, padding='valid')(m)
m = layers.advanced_activations.ELU(.3)(m)
m = layers.GlobalAveragePooling1D(name='embedding')(m)
m = layers.Dense(32)(m)
m = layers.advanced_activations.ELU(.3)(m)
m = layers.Dense(16)(m)
m = layers.advanced_activations.ELU(.3)(m)
m = layers.Dense(16)(m)
m = layers.advanced_activations.ELU(.3)(m)
ae_out = layers.Dense(X.shape[-1])(m)
model = models.Model(ae_in, ae_out)
model.summary()
model.compile(optimizers.Adam(.0001), loss='mse')

model.fit(X[:, :-2], X[:, -1],
          batch_size=64,
          epochs=5)
