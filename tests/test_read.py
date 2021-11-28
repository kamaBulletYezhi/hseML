import numpy as np
from src.data.readdataset import read_cancer_dataset
from src.data.readdataset import read_spam_dataset
import pytest
from src.basedir import BASE_DIR

eps = 10**(-6)


class TestReadCancer:
    dsp = BASE_DIR+"/data/raw/cancer.csv"
    X, y = read_cancer_dataset(dsp)

    def test_cancer_mean(self):
        X = TestReadCancer.X
        assert (np.abs(X.mean(0)) < eps).all()

    def test_cancer_std(self):
        X = TestReadCancer.X
        assert (np.abs(X.std(0) - 1) < eps ** 0.5).any()


class TestReadSpam:
    dsp = BASE_DIR+"/data/raw/spam.csv"
    X, y = read_spam_dataset(dsp)

    def test_spam_mean(self):
        X = TestReadSpam.X
        assert (np.abs(X.mean(0)) < eps).all()

    def test_spam_std(self):
        X = TestReadSpam.X
        assert (np.abs(X.std(0) - 1) < eps ** 0.5).any()


