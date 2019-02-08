import numpy as np
import pandas as pd
from keras import models, layers, optimizers
import glob


import nn_util


nn_util.limit_keras_gpu_mem(.1)
np.random.seed(11798)  # So that data splitting is the same each run.



print('Loading unsupervised data')
folderList = ['AAA', 'BBB','CCC', 'DDD','EEE', 'FFF', 'GGG']
XList = list()
pidList = list()
unique_pids = list()
train_pids = list()
test_pids = list()
val_pids = list()
train_pids = list()
val_Xs = list()
train_Xs = list()
unique_pid = list()
train_pid = list()
test_pid = list()
val_pid = list()
train_pid = list()
val_X = list()
train_X = list()
for i in folderList:
    for np_name in glob.glob('processed_data/seq/' + i + '/seq_X-unsup-*.npy'):
        data = np.load(np_name)
        if data.shape[1] == 3:
            XList.append(data)
    for np_name in glob.glob('processed_data/seq/' + i + '/seq_pids-unsup-*.npy'):
        pidList.append(np.load(np_name))
    # Select some participants to use for training, validation, and testing sets.
    pids = np.concatenate(pidList)
    X = np.concatenate(XList, axis=0)
    unique_pid = np.unique(pids)
    np.append(unique_pids,unique_pid)
    train_pid = np.random.choice(unique_pid, int(len(unique_pid) * .8), replace=False)
    test_pid = unique_pid[np.isin(unique_pid, train_pid, invert=True)]
    np.append(test_pids, test_pid)
    # test_X = X[np.isin(pids, train_pids, invert=True)]  # Don't actually need testing data for AE.
    # Further split training data into train/validation sets.
    val_pid = np.random.choice(train_pid, int(len(train_pid) * .2), replace=False)
    np.append(val_pids,val_pid)
    train_pid = np.setdiff1d(train_pid, val_pid)
    np.append(train_pids, train_pid)
    print(pids)
    print(val_pid.shape)
    val_X = X[np.isin(pids, val_pid)]
    np.append(val_Xs, val_X)
    train_X = X[np.isin(pids, train_pid)]
    np.append(train_Xs, train_X)
    assert len(train_pid) + len(val_pid) + len(test_pid) == len(unique_pid)

unique_pids = np.concatenate(unique_pids)
train_pids = np.concatenate(train_pids)
test_pids = np.concatenate(test_pids)
val_pids = np.concatenate(val_pids)
train_pids = np.concatenate(train_pids)
val_Xs = np.concatenate(val_Xs)
train_Xs = np.concatenate(train_Xs)

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

model.fit(train_X[:, :-2], train_X[:, -1],
          validation_data=(val_X[:, :-2], val_X[:, -1]),
          batch_size=64,
          epochs=5)

print('Loading session-level supervised data')
for i in folderList:

    X = np.load(glob.glob('processed_data/seq_X-sess.npy'))# X is an array of 2D sequences, not a 3D tensor.
    y = np.load(glob.glob('processed_data/seq_y-sess.npy'))
    pids = np.load(glob.glob('processed_data/seq_pids-sess.npy'))
    print(str(len(y)) + ' sequences loaded')
    '''
    for i in y:
        X = X[np.invert(np.isnan(i))]
        pids = pids[np.invert(np.isnan(i))]
        y = y[np.invert(np.isnan(i))]
    y = y / 6  # Rescale grade [0, 1].
    '''
    train_X, val_X, test_X = X[np.isin(pids, train_pids)], X[np.isin(pids, val_pids)], \
        X[np.isin(pids, test_pids)]
    train_y, val_y, test_y = y[np.isin(pids, train_pids)], y[np.isin(pids, val_pids)], \
        y[np.isin(pids, test_pids)]
    assert len(train_X) + len(val_X) + len(test_X) == len(X)
    print(len(train_y) + len(val_y) + len(test_y))
    assert len(train_y) + len(val_y) + len(test_y) == len(y)
    print('Train on %d, validate on %d, test on %d' % (len(train_y), len(val_y), len(test_y)))

def variable_gen(x_arr, y_arr=None):
    # Data are variable-length sequences, so we must either pad or feed manually.
    while True:
        for i in range(len(x_arr)):
            if y_arr is None:
                yield np.expand_dims(x_arr[i], 0)  # Batches of size 1, no y (for prediction).
            else:
                yield np.expand_dims(x_arr[i], 0), np.expand_dims(y_arr[i], 0)

for l in model.layers:
    l.trainable = False
embedding_layer = [l for l in model.layers if l.name == 'embedding'][0]
sup = layers.Dense(16)(embedding_layer.output)
sup = layers.advanced_activations.ELU(.3)(sup)
sup = layers.Dense(16)(sup)
sup = layers.advanced_activations.ELU(.3)(sup)
sup_out = layers.Dense(1)(sup)
sup_model = models.Model(ae_in, sup_out)
sup_model.summary()
sup_model.compile(optimizers.Adam(.0001), loss='mse')

sup_model.fit_generator(variable_gen(train_X, train_y),
                        validation_data=variable_gen(val_X, val_y),
                        validation_steps=len(val_X),
                        steps_per_epoch=len(train_X),
                        epochs=100)

# Compare predictions to baselines.
preds = sup_model.predict_generator(variable_gen(val_X), steps=len(val_X)).T
print('MSE of validation baseline (mean y): ' + str(((val_y - val_y.mean()) ** 2).mean()))
print('Point biserial r for validation data: ' + str(np.corrcoef(val_y, preds)[0, 1]))
preds = sup_model.predict_generator(variable_gen(train_X), steps=len(train_X)).T
print('MSE of train baseline (mean y): ' + str(((train_y - train_y.mean()) ** 2).mean()))
print('Point biserial r for train data: ' + str(np.corrcoef(train_y, preds)[0, 1]))

# Should not look at test set performance until model is finalized.
# preds = sup_model.predict_generator(variable_gen(test_X), steps=len(test_X)).T
# print('MSE of test: ' + str(((preds - test_y) ** 2).mean()))
# print('MSE of test baseline (mean y): ' + str(((test_y - test_y.mean()) ** 2).mean()))
# print('Pearson\'s r for test set: ' + str(np.corrcoef(test_y, preds)[0, 1]))
