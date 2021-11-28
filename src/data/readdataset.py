import numpy as np
import pandas as pd
from typing import Tuple
'''
Read and randomise dataset  
'''


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    df = pd.read_csv(path_to_csv, header=0).sample(frac=1, ignore_index=True)
    X = df.drop(labels='label', axis=1)
    y = df.label.apply(lambda s: 1 if s == 'M' else 0)
    X -= X.mean(0)
    X /= X.std(0)
    return np.array(X), np.array(y)


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    df = pd.read_csv(path_to_csv, header=0).sample(frac=1, ignore_index=True)
    X = df.drop(labels='label', axis=1)
    y = df['label']
    X -= X.mean(0)
    X /= X.std(0)
    return np.array(X), np.array(y)
