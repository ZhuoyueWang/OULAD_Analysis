# Contains utility functions for neural network code, with documentation for each.
from collections import OrderedDict

import numpy as np
from keras import callbacks, backend as K
from sklearn.metrics import cohen_kappa_score, roc_auc_score, matthews_corrcoef, precision_score
import tensorflow as tf


def batch_transpose2d(x):
    """Transpose input in a network. Useful for applying TimeDistributed layers across features
    instead of across time, for example. Use in Lambda like layers.Lambda(batch_transpose2d).

    Args:
        x (Keras Tensor): tensor with shape (BATCH_SIZE, SEQUENCE_LENGTH, NUM_FEATURES)

    Returns:
        Keras Tensor: Batch tensor with each sample sub-tensor transposed
    """
    return K.map_fn(K.transpose, x)


def eval_predictions(y_true, y_pred, id_dict={}):
    """Evaluate predictions for Cohen's kappa, MCC, AUC, and some other info. Predictions must be
    for one class and in the range [0, 1] for kappa and MCC calculation.

    Args:
        y_true (arr): Ground truth labels in a 1D Numpy array or list
        y_pred (arr): Predicted labels in a 1D Numpy array or list
        id_dict (dict, optional): Dictionary to be prepended on the result, to help keep track of
            multiple results

    Returns:
        collections.OrderedDict: result metrics
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert(len(y_true.shape) == 1 and len(y_pred.shape) == 1)  # Verify input shape.
    assert(set(np.unique(y_true)) == {0, 1})  # Verify 0-1 labels.
    y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())  # Scale predictions to [0, 1].
    result = OrderedDict(id_dict)
    result['kappa'] = cohen_kappa_score(y_true, y_pred > .5)
    result['auc'] = roc_auc_score(y_true, y_pred)
    result['mcc'] = matthews_corrcoef(y_true, y_pred > .5)
    result['precision'] = precision_score(y_true, y_pred > .5)
    result['data_imbalance'] = sum(y_true) / len(y_true)
    result['prediction_imbalance'] = sum(y_pred > .5) / len(y_pred)
    # Optimize threshold for kappa.
    best_kappa = -1
    best_thresh = 0
    for thresh in range(100):
        kappa = cohen_kappa_score(y_true, y_pred > thresh / 100)
        if kappa > best_kappa:
            best_kappa = kappa
            best_thresh = thresh / 100
    result['max_kappa'] = best_kappa
    result['max_kappa_thresh'] = best_thresh
    result['max_kappa_imbalance'] = sum(y_pred > best_thresh) / len(y_pred)
    return result


def one_hot_kappa(y_true, y_pred, hot_column=1):
    '''
    Calculate Cohen's kappa from one-hot encoded continuous predictions. Scales predictions of the
    positive class to [0, 1] and thresholds at .5, using only the output indexed by hot_column. Use
    functools.partial to set hot_column if needed to pass the function by reference.

    :param y_true: Ground truth, with shape (?, 2)
    :param y_pred: Predictions with shape (?, 2)
    :param hot_column: Kappa will be calculated from this output column, i.e. y_pred.T[hot_column]
    :returns: Cohen's kappa
    '''
    y_pred, y_true = y_pred.T[hot_column], y_true.T[hot_column]
    y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
    y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min())
    return cohen_kappa_score(y_true > .5, y_pred > .5)


def one_hot_auc(y_true, y_pred, hot_column=1):
    '''
    Calculate area under the receiver operating characteristic curve (AUC). Ignores the negative
    class column of predictions (column index 0). Use functools.partial to set hot_column if needed
    to pass the function by reference.

    :param y_true: Ground truth, with shape (?, 2)
    :param y_pred: Continuous predictions with shape (?, 2)
    :param hot_column: Kappa will be calculated from this output column, i.e. y_pred.T[hot_column]
    :returns: AUC
    '''
    return roc_auc_score(y_true.T[hot_column], y_pred.T[hot_column])


class MetricCallback(callbacks.Callback):
    def __init__(self, x, y_true, metric=one_hot_kappa,
                 best_checkpoint_filename=None, verbose_label=None):
        '''
        Calculate a custom metric at the end of each epoch, and optionally save a model checkpoint
        whenever a new maximum value of that metric is reached.

        :param x: Input data to use for prediction
        :param y_true: Ground truth for the predictions
        :param metric: Function taking y_true, y_pred parameters and returning a number
        :param best_checkpoint_filename: If specified, a model checkpoint will be saved when a new
                                         maximum value of the metric function is returned
        :param verbose_label: If specified (e.g., ', kappa=') then the metric value will be printed
        '''
        super(MetricCallback, self).__init__()
        self.fname = best_checkpoint_filename
        self.x = x
        self.y = y_true
        self.metric = metric
        self.best_perf = -1
        self.label = verbose_label

    def on_epoch_end(self, epoch, logs={}):
        perf = self.metric(self.y, self.model.predict(self.x))
        if self.label:
            print(self.label + str(perf)[:6] + (' (new best)' if perf >= self.best_perf else ''))
        if perf >= self.best_perf:  # Prefer models from later epochs even if equal performance.
            self.best_perf = perf
            if self.fname:
                self.model.save(self.fname)


def limit_keras_gpu_mem(preallocate_prop=.20):
    '''
    Limit the amount of GPU RAM that is pre-allocated for models, so that more than two models can
    be simultaneously trained. Assumes TF backend.

    :param preallocate_prop: Proportion of GPU RAM to pre-allocate
    '''
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = preallocate_prop
    K.tensorflow_backend.set_session(tf.Session(config=config))


def one_hot_encode(labels, hot=1, cold=0, one_vs_other=None):
    """One-hot encode labels, which will also be assigned sequential integers in sorted() order.

    Args:
        labels ([iterable]): Labels may be integers, strings or otherwise
        hot (int, optional): Value to use for the "hot" code
        cold (int, optional): Value to use for the "cold" code
        one_vs_other (label, optional): If specified, this class is "hot" and all others are "cold"

    Returns:
        (Numpy array, Numpy array): Tuple of (one-hot encoded labels, sequential integer labels)
    """
    assert(len(np.shape(labels)) == 1)  # Verify input is 1D.
    if one_vs_other:
        labels = [1 if l == one_vs_other else 0 for l in labels]
    label_map = {label: index for index, label in enumerate(sorted(set(labels)))}
    y = np.full(shape=[len(labels), len(label_map)], fill_value=cold)
    int_y = np.zeros(len(labels))
    for i, l in enumerate(labels):
        y[i][label_map[l]] = hot
        int_y[i] = label_map[l]
    return y, int_y


def batch_sample_counts(label_arr, batch_size, oversampling_factor=.5):
    '''
    Calculate how many instances of each class should be present in a batch, given the priors of
    each class and a desired level of oversampling for minority classes. oversampling_factor allows
    creation of batches that oversample minority classes to different degrees, because no
    oversampling often leads to models predicting only the majority class but equalizing base rates
    often lead to excessive over-prediction of the minority class.

    :param label_arr: Iterable of consecutive integer labels
    :param batch_size: Integer size of batches, which will influence how much rounding occurs
    :param oversampling_factor: Proportion of the distance between the base rate of minority classes
                                and equal base rates to be covered (0 means no oversampling, 1 means
                                all classes will be equally represented)
    '''
    assert(oversampling_factor <= 1)  # Oversampling above equal rate not supported.
    props = {l: sum(1 for v in label_arr if v == l) / len(label_arr) for l in set(label_arr)}
    new_props = {}
    for label in props:
        new_props[label] = props[label] + (1 / len(props) - props[label]) * oversampling_factor
    assert(abs(1 - sum(new_props.values())) < .0001)  # Make sure proportions still add to 1.
    counts = {l: max(1, round(new_props[l] * batch_size)) for l in new_props}  # >=1 instance/class.
    while sum(counts.values()) > batch_size:  # Too many instances, we need to cut back.
        majority_class = [l for l in counts if counts[l] == max(counts.values())][0]
        counts[majority_class] -= 1
    while sum(counts.values()) < batch_size:  # Not enough instances, add more.
        minority_class = [l for l in counts if counts[l] == min(counts.values())][0]
        counts[minority_class] += 1
    assert(sum(counts.values()) == batch_size)  # Check that the result has the right size.
    return counts


def oversample_classes(arr, labels, new_base_rates='auto'):
    '''
    Randomly over-sample instances as needed to achieve a specified base rate for each class. At
    most N-1 of the N classes will be oversampled. In other words, oversampling may not perfectly
    match the specified new base rates if doing so would require oversampling from every class to
    create a dataset with sufficient instances to be evenly divisible by the target base rates.

    :param arr: Numpy array of input instances (i.e. training X)
    :param labels: Labels corresponding to instances, either consecutive numbers or one-hot encoded
    :param new_base_rates: One base rate per class, or 'auto' to set to 1/(number of classes) for
                           each class
    :returns: Tuple of Numpy arrays (arr, labels) with oversampling performed
    '''
    # Convert one-hot to categorical if needed.
    one_hot = len(np.shape(labels)) == 2
    if len(np.shape(labels)) != 1 and not one_hot:
        raise ValueError('labels should be consecutive numbers or one-hot encoded')
    if new_base_rates != 'auto' and sum(new_base_rates) != 1:
        raise ValueError('new_base_rates need to sum to 1')
    if one_hot:
        labels = [[i for i, x in enumerate(l) if x == 1][0] for l in labels]
    try:
        n_classes = len(np.unique(labels))
    except TypeError:
        raise TypeError('labels should be consecutive integers')
    labels = list(labels)
    if len(set(labels) - set(range(n_classes))) != 0:
        raise ValueError('categorical labels should be consecutive integers starting at 0')

    # Set auto base rate (all equal) if requested.
    if new_base_rates == 'auto':
        new_base_rates = [1. / n_classes] * n_classes

    # Iteratively over-sample all classes that are under target rates until all are reached.
    adjusted_classes = set()
    class_counts = [sum([y == i for y in labels]) for i in range(n_classes)]
    while True:
        under_classes = [i for i in range(n_classes)
                         if class_counts[i] / sum(class_counts) < new_base_rates[i]]
        if len(under_classes) == 0:
            break  # No more adjustments need to be made, base rates are perfectly met.
        adjusted_classes.update(under_classes)
        if len(adjusted_classes) == len(class_counts):
            break  # No more adjustments should be made, we have adjusted all minority classes.
        for i in under_classes:
            class_counts[i] += 1

    # Actually do the over-sampling all at once at the end for speed.
    new_i = []
    for c in range(n_classes):
        y_i = [i for i, x in enumerate(labels) if x == c]
        to_sample = class_counts[c] - len(y_i)
        if to_sample > 0:
            labels.extend([c] * to_sample)
            new_i.extend(np.random.choice(y_i, to_sample, replace=True))
    if len(adjusted_classes) > 0:
        arr = np.concatenate([arr, arr[new_i]])

    # Convert back to one-hot if needed.
    if one_hot:
        labels = [[1 if l == i else 0 for i in range(n_classes)] for l in labels]

    return arr, np.array(labels)


def z_standardize(train_pandas_df, test_pandas_df=None, columns=None, clip_magnitude=None):
    """
    Shift and rescale data to have mean = 0 and standard deviation = 1, based on mean/sd computed
    from training data only (to prevent train/test set leakage). Missing values are ignored.

    :param train_pandas_df: Training data as a pandas DataFrame
    :param test_pandas_df: Testing data as a pandas DataFrame (optional)
    :param columns: List of columns to standardize (typically in + out columns in the network); if
                    none, then all columns will be standardized
    :param clip_magnitude: Clip values beyond clip_magnitude standard deviations (i.e. Winsorize)
    :returns: Tuple of (train, test) DataFrame copies with "columns" standardized, or only train if
              test_pandas_df was not specified
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy() if test_pandas_df is not None else None
    for col in columns if columns is not None else train_pandas_df:
        train_m = train[col].mean()
        train_sd = train[col].std()
        train[col] = (train[col] - train_m) / train_sd
        if test is not None:
            test[col] = (test[col] - train_m) / train_sd
        if clip_magnitude is not None:
            train[col] = [min(max(v, -clip_magnitude), clip_magnitude) for v in train[col]]
            if test is not None:
                test[col] = [min(max(v, -clip_magnitude), clip_magnitude) for v in test[col]]
    return (train, test) if test is not None else train


def rescale(train_pandas_df, test_pandas_df=None, columns=None, new_min=0, new_max=1):
    """
    Shift and rescale data to have min = new_min and max = new_max, based on min and max computed
    from training data only (to prevent train/test set leakage). Missing values are ignored.

    :param train_pandas_df: Training data as a pandas DataFrame
    :param test_pandas_df: Testing data as a pandas DataFrame (optional)
    :param columns: List of columns to standardize (typically in + out columns in the network); if
                    none, then all columns will be standardized
    :param new_min: New minimum in the rescaled output
    :param new_max: New maximum in the rescaled output
    :returns: Tuple of (train, test) DataFrame copies with "columns" rescaled, or only train if
              test_pandas_df was not specified
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy() if test_pandas_df is not None else None
    for col in columns if columns is not None else train_pandas_df:
        train_min = train[col].min()
        train_range = train[col].max() - train_min
        train[col] = (train[col] - train_min) / train_range * (new_max - new_min) + new_min
        if test is not None:
            test[col] = (test[col] - train_min) / train_range * (new_max - new_min) + new_min
    return (train, test) if test is not None else train


def make_sequences(pandas_df,
                   in_columns,
                   participant_id_col=None,
                   sequence_len=10,
                   min_valid_prop=.7,
                   missing_fill=0,
                   overlap=10,   #10
                   verbose=False):
    """
    Create sequences of data of a specific length using a sliding window that moves by "overlap".
    Sequences are necessary for training sequential models like LSTMs/GRUs. Also finds the indices
    of data points that can possibly be predicted, so targets (i.e. y) can be extracted.

    :param pandas_df: Pandas DataFrame object with sequential data
    :param in_columns: List of columns to include in the resulting input sequences (i.e., X)
    :param participant_id_col: If specified, sequences will not overlap different participant IDs;
                               i.e., no sequence will include data from more than one participant
    :param sequence_len: Number of rows to include in each sequence
    :param min_valid_prop: Minimum proportion of non-missing data points per sequence; the sequence
                           will not be included in the result if this constraint is not met
    :param missing_fill: Replace missing values with this (iff min_valid_prop is satisfied)
    :param overlap: Number of rows to move the sliding window forward by (usually 1)
    :param verbose: Print the number of sequences created every 1000 sequences
    :returns: Tuple of (ndarray of sequences [i.e. inputs], list of target indices)
    """
    seqs = []
    indices = []
    for i in range(sequence_len, len(pandas_df) + 1, overlap):
        seq = pandas_df.iloc[i - sequence_len:i][in_columns]
        if participant_id_col and \
                len(pandas_df.iloc[i - sequence_len:i][participant_id_col].unique()) > 1:
            continue  # Cannot have sequences spanning multiple participant IDs.
        if seq.count().sum() / float(sequence_len * len(in_columns)) < min_valid_prop:
            continue  # Not enough valid data in this sequence.
        seqs.append(seq.fillna(missing_fill).values)
        indices.append(i - 1)
        if verbose and i / overlap % 1000 == 0:
            print('%.1f%%' % (i / len(pandas_df) * 100), end='\r')
    return np.array(seqs), indices


if __name__ == '__main__':
    # Do some testing.
    import pandas as pd
    df = pd.DataFrame.from_records([{'pid': 'p1', 'a': 1, 'b': 2, 'c': 'x'},
                                    {'pid': 'p1', 'a': 4.5},
                                    {'pid': 'p2', 'a': 2, 'b': 1, 'c': 'y'},
                                    {'pid': 'p2', 'a': 3, 'b': 0, 'c': 'x'},
                                    {'pid': 'p2', 'a': 4, 'b': 0, 'c': 'z'}])
    print(df)
    print('Sequences with strict min_valid_prop requirement:')
    print(make_sequences(df, ['a', 'b'], sequence_len=3, min_valid_prop=.9))
    print('Sequences with less strict min valid data:')
    print(make_sequences(df, ['a', 'b'], sequence_len=3, min_valid_prop=.5))
    print('Sequences with string column:')
    print(make_sequences(df, ['c'], sequence_len=3, min_valid_prop=.5, missing_fill='_'))
    print('Sequences bounded by participant id:')
    print(make_sequences(df, ['a'], participant_id_col='pid', sequence_len=2))

    print('Standardized with pid=p2 as train, pid=p1 as test:')
    a, b = z_standardize(df[df.pid == 'p2'], df[df.pid == 'p1'], ['a', 'b'])
    print(a)
    print(b)

    print('Rescaled to [-1, 1] with pid=p2 as train, pid=p1 as test:')
    a, b = rescale(df[df.pid == 'p2'], df[df.pid == 'p1'], ['a', 'b'], new_min=-1, new_max=1)
    print(a)
    print(b)

    print('Oversampling to equal rates with [0, 1, 0, 2, 2] as class labels')
    x, y = oversample_classes(df.values, [0, 1, 0, 2, 2])
    print(x)
    print(y)
    print('Oversampling to [.4, .2, .4] rates with [0, 1, 0, 2, 2] as class labels (no change)')
    x, y = oversample_classes(df.values, [0, 1, 0, 2, 2], [.4, .2, .4])
    print(x)
    print(y)
    print('Oversampling to [.4, .3, .3] rates with [0, 1, 0, 2, 2] as class labels')
    x, y = oversample_classes(df.values, [0, 1, 0, 2, 2], [.4, .3, .3])
    print(x)
    print(y)
    print('Oversampling to [.1, .45, .45] rates with [0, 1, 0, 2, 2] as class labels')
    x, y = oversample_classes(df.values, [0, 1, 0, 2, 2], [.10, .45, .45])
    print(x)
    print(y)

    print('One-hot encoding [a, a, b, c, a, b, c]')
    y, int_y = one_hot_encode([l for l in 'aabcabc'])
    print(y)
    print(int_y)
    print('One-vs-other encoding [a, a, b, c, a, b, c] with b as 1 and others as -1')
    y, int_y = one_hot_encode([l for l in 'aabcabc'], 1, -1, 'b')
    print(y)
    print(int_y)

    print('Batch sample counts for {0: 80, 1: 15, 2: 5} priors, batch size 9, and oversampling')
    y = [0 if i < 80 else 1 if i < 95 else 2 for i in range(100)]
    print(batch_sample_counts(y, 9, 1.0))
    print('Batch sample counts with batch size 10 and no oversampling')
    print(batch_sample_counts(y, 10, 0))
    print('Batch sample counts with batch size 3')
    print(batch_sample_counts(y, 3))
    print('Batch sample counts with batch size 100 and half oversampling')
    print(batch_sample_counts(y, 100, .5))

    print('\nTransposing [[[11, 12], [21, 22]], [[31, 32], [41, 42]]] in a Keras graph')
    x = K.variable(np.array([[[11, 12], [21, 22]], [[31, 32], [41, 42]]]))
    print(K.eval(batch_transpose2d(x)))
