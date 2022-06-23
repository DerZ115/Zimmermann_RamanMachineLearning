import logging
import numpy as np
import pandas as pd
from pybaselines.misc import beads
from pybaselines.morphological import mormol, rolling_ball
from pybaselines.whittaker import arpls, asls
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import savgol_filter, find_peaks

prep_logger = logging.getLogger("__main__." + __name__)

class BaselineCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, method="asls"):
        prep_logger.debug("Creating instance of BaselineCorrector")
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        prep_logger.info("Starting baseline correction")
        prep_logger.info(f"Using baseline method {self.method}")
        X = X.copy()
        if type(X) != np.ndarray:
            X = np.asarray(X)

        bl = np.zeros_like(X)

        if self.method == "asls":
            for i, row in enumerate(X):
                bl[i] = asls(row)[0]

        elif self.method == "arpls":
            for i, row in enumerate(X):
                bl[i] = arpls(row)[0]

        elif self.method == "mormol":
            for i, row in enumerate(X):
                bl[i] = mormol(row)[0]

        elif self.method == "rolling ball":
            for i, row in enumerate(X):
                bl[i] = rolling_ball(row)[0]

        elif self.method == "beads":
            for i, row in enumerate(X):
                bl[i] = beads(row)[0]

        else:
            prep_logger.critical("Invalid baseline method")
            raise ValueError(f"Method {self.method} does not exist.")

        prep_logger.info("Finished baseline correction")
        return X - bl


class ColumnSelectorPCA(BaseEstimator, TransformerMixin):
    """Class to select a range of components from PCA, so that components do not 
    have to be calculated over and over."""

    def __init__(self, n_components=None):
        """Initialize range of components."""
        if n_components == None:
            n_components = -1
        self.n_components = n_components

    def fit(self, X, y=None):
        """Does not do anything, included for compatibility with pipelines"""
        return self

    def transform(self, X, y=None):
        """Return the previously selected range of components."""
        return X[:, 0:self.n_components]


class RangeLimiter(BaseEstimator, TransformerMixin):
    def __init__(self, lim=(None, None), reference=None):
        prep_logger.debug("Creating instance of RangeLimiter")
        self.lim = lim
        self.reference = reference

    def fit(self, X, y=None):
        self.lim = list(self.lim)
        prep_logger.debug("Range Limiter: Validating parameters")
        self._validate_params(X)

        if self.reference is not None:
            self.lim_ = [
                np.where(self.reference >= self.lim[0])[0][0],
                np.where(self.reference <= self.lim[1])[0][-1] + 1
            ]
        else:
            self.lim_ = [self.lim[0], self.lim[1] + 1]

        return self

    def transform(self, X, y=None):
        prep_logger.info(f"Reducing spectral range to {self.lim}")
        return X[:, self.lim_[0]:self.lim_[1]]

    def _replace_nones(self, X):
        if self.lim[0] is None:
            prep_logger.debug("Lower limit is None, replacing...")
            if self.reference is None:
                self.lim[0] = 0
            else:
                self.lim[0] = self.reference[0]

        if self.lim[1] is None:
            prep_logger.debug("Upper limit is None, replacing...")
            if self.reference is None:
                self.lim[1] = X.shape[1]
            else:
                self.lim[1] = self.reference[-1]

    def _validate_params(self, X):
        self.reference = np.asarray(self.reference)
        if len(self.lim) != 2:
            prep_logger.info("Wrong number of values for lim")
            raise ValueError("Wrong number of values for lim.")

        if self.reference is not None and \
                np.any(self.reference[:-1] > self.reference[1:]):
            prep_logger.critical("Reference array is not sorted")
            raise ValueError("Reference array is not sorted.")

        self._replace_nones(X)

        if np.any([
            self.reference is None and (
                self.lim[0] < 0 or self.lim[1] > X.shape[1]),
            self.reference is not None and (
                self.lim[0] < self.reference[0] or self.lim[1] > self.reference[-1])]):
            prep_logger.critical("Given index is out of range")
            raise IndexError(
                "Index out of range. Please check the provided indices.")


class SavGolFilter(BaseEstimator, TransformerMixin):
    """Class to smooth spectral data using a Savitzky-Golay Filter."""

    def __init__(self, window=15, poly=3):
        """Initialize window size and polynomial order of the Savitzky-Golay Filter"""
        prep_logger.debug("Creating instance of SavGolFilter")
        self.window = window
        self.poly = poly

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        prep_logger.info(f"Applying Savitzky-Golay Filter with window size {self.window} and polynomial order {self.poly}")
        X_smooth = savgol_filter(
            X, window_length=self.window, polyorder=self.poly)
        prep_logger.info("Smoothing complete")
        return (X_smooth.T - X_smooth.min(axis=1)).T


class PeakPicker(BaseEstimator, TransformerMixin):
    def __init__(self, min_dist=None):
        prep_logger.debug("Creating instance of PeakPicker")
        self.min_dist = min_dist

    def fit(self, X, y=None):
        prep_logger.info("Finding peaks in the mean spectrum")
        X_mean = X.mean(axis=0)
        self.peaks_ = find_peaks(X_mean, distance=self.min_dist)[0]
        prep_logger.debug(f"Number of peaks found: {self.peaks_}")
        return self

    def transform(self, X, y=None):
        prep_logger.info("Selecting previously identified peaks")
        return X[:, self.peaks_]
