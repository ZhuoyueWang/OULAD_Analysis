# Create sequences from the action logs to be used for training unsupervised sequence models.
import pandas as pd
import numpy as np
from tqdm import tqdm

import nn_util


SEQ_LEN = 10

print('Loading data')
df = pd.read_csv('processed_data/vle_info.csv')

print('Applying distribution transformations')
