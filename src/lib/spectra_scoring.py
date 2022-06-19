import numpy as np
import time
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import argrelmin, savgol_filter

from .preprocessing import BaselineCorrector


def limit_range(data, limit_low, limit_high):
    """Limit the frequency range of spectral data.

    Args:
        data (pd.DataFrame): Raw spectral data, with each row representing one spectrum. IMPORTANT: Make sure that the column names (i.e. the frequencies/wavenumbers) are of type float or int.
        limit_low (int or float or None): Lower limit of the spectral range.
        limit_high (int or float or None): Upper limit of the spectral range.

    Returns:
        pd.DataFrame: Data with reduced range.
    """
    if limit_low == None:
        limit_low = data.columns[0]

    if limit_high == None:
        limit_high == data.columns[-1]

    return data.loc[:, limit_low:limit_high]


def baseline_correction(data, method="asls"):
    """Estimate and subtract the baseline from a set of spectra.

    Args:
        data (pandas.DataFrame): Raw spectral data. Rows represent the individual spectra.
        method (str, optional): Baseline correction method to use. Defaults to "asls".

    Returns:
        pandas.DataFrame: Baseline-corrected spectra.
    """
    wns = data.columns

    bl = BaselineCorrector(method=method)
    data = bl.fit_transform(data)
    data = pd.DataFrame(data, columns=wns)
    return data


def peakRecognition(data, sg_window, threshold):
    """Determines the number of peaks in each spectrum based on a 2nd derivative Savitzky-Golay-Filter.

    Args:
        data (pd.DataFrame): Baseline corrected spectra
        sg_window (int): Window width of the Savitzky-Golay-Filter (must be odd)

    Returns:
        peaks (list): List of lists with the peaks found in each spectrum
    """

    wns = data.columns.astype("float64")
    data_sg = pd.DataFrame(
        savgol_filter(data, window_length=2*sg_window+1, polyorder=3, deriv=2), columns=wns)

    peaks = []

    for i, row in data_sg.iterrows():
        row_peaks = argrelmin(row.values)[0]
        row_peaks = [peak for peak in row_peaks if row.iloc[peak]
                     < -threshold]  # Remove peaks below threshold

        # Combine peaks w/o positive 2nd derivative between them
        peak_condensing = []
        peaks_condensed = []
        for j in range(len(row)):
            if j in row_peaks:
                peak_condensing.append(j)
            if row.iloc[j] > 0 and len(peak_condensing) > 0:
                peaks_condensed.append(int(np.mean(peak_condensing)))
                peak_condensing = []
        if len(peak_condensing) > 0:
            peaks_condensed.append(int(np.mean(peak_condensing)))

        peaks.append(peaks_condensed)

    return peaks


def calc_scores(data, peaks, score_measure, n_peaks_influence):
    """Calculates the quality scores for each spectrum

    Args:
        data (pandas.DataFrame): Baseline-corrected spectra.
        peaks (list): List of lists with the peaks found in each spectrum.
        score_measure (int): Sets intensity measure used for score calculation.
        n_peaks_influence (int): Sets influence of peak number on the score. 
        detailed (bool): Whether the individual parts of the score (height/area and number of peaks) should be returned. Default: False

    Returns:
        scores_peaks (list): Overall score for each spectrum.
        scores (list): Pure intensity score for each spectrum (w/o number of peaks). Only if detailed=True
        n_peaks_all (list): Number of peaks in each spectrum. Only if detailed=True
    """

    scores = []
    n_peaks_all = []

    for i, row in enumerate(peaks):
        n_peaks = len(row)
        if n_peaks == 0:
            score = 0
        elif score_measure == 0:
            score = 1
        elif score_measure == 1:  # median height
            heights = [data.iloc[i, k] for k in row]
            score = np.median(heights)
        elif score_measure == 2:  # mean height
            heights = [data.iloc[i, k] for k in row]
            score = np.mean(heights)
        elif score_measure == 3:  # mean area
            score = simpson(data.iloc[i, :], data.columns) / n_peaks
        elif score_measure == 4:  # total area
            score = simpson(data.iloc[i, :], data.columns)

        scores.append(score)
        n_peaks_all.append(n_peaks)

        if n_peaks == 0:
            scores_peaks = 0
        elif n_peaks_influence == 0:
            scores_peaks = scores
        elif n_peaks_influence == 1:
            scores_peaks = [n*score for n, score in zip(n_peaks_all, scores)]
        elif n_peaks_influence == 2:
            scores_peaks = [score**(n/50)
                            for n, score in zip(n_peaks_all, scores)]

    n_peaks_all = [n_peaks for _, n_peaks in sorted(
        zip(scores_peaks, n_peaks_all))]
    n_peaks_all.reverse()
    return scores_peaks, scores, n_peaks_all


def sort_spectra(data, scores):
    """Group spectra by class, sort each class by the quality score and optionally only retain the n highest quality spectra.

    Args:
        data (pandas.DataFrame): Original (not baseline-corrected) data. 
        labels (pandas.Series): Class labels of the spectral data
        scores (list): Quality score of each spectrum
        n (int, optional): Number of spectra to be used from each class. If None, all spectra will be retained. Defaults to None.

    Returns:
        pandas.DataFrame: Sorted data, Class labels are included as the first column.
    """
    data_sorted = data.copy()
    data_sorted.insert(0, "score", scores)
    data_sorted = data_sorted.sort_values("score", ascending=False)
    data_sorted = data_sorted.reset_index(drop=True)

    return data_sorted


def remove_low_quality(data, n=None, min_n=0, min_score=0):
    if min_score == 0 and min_n != 0:
        raise ValueError("min_n only works in combination with min_score")

    data_out = pd.DataFrame(columns=data.columns)

    if n is not None:
        for _, group in data.groupby("label"):
            data_out = pd.concat([data_out, group.iloc[:n, :]])
        
    elif min_score != 0:
        for _, group in data.groupby("label"):
            group_out = group.loc[group.score >= min_score, :]
            if len(group_out) < min_n:
                group_out = group.iloc[:min_n, :]
            data_out = pd.concat([data_out, group_out])

    else:
        data_out = data

    data_out = data_out.reset_index(drop=True)
    return data_out



def score_sort_spectra(data,
                       n=None,
                       min_n=0,
                       min_score=0,
                       limits=(None, None),
                       bl_method="asls",
                       sg_window=17,
                       threshold=0.5,
                       score_measure=1,
                       n_peaks_influence=1,
                       detailed=False):
    """Convenience function for baseline-correcting, scoring and sorting spectral data.

    Args:
        data (pandas.DataFrame): Spectral data with each row representing a spectrum. Class labels (or similar) should be in a column named 'label'
        n (int, optional): Number of spectra to retain from each class. If None, all spectra will be kept. Defaults to None.
        limits (tuple, optional): If the spectral range should be reduced, the lower and upper limits of the spectral range. Defaults to (None, None).
        bl_method (str, optional): Baseline correction method to use. Defaults to "asls".
        sg_window (int, optional): Window half-width for the Savitzky-Golay filter. Defaults to 17.
        threshold (float, optional): Threshold value for the second derivative. Potential peaks must have a lower (negative) value than this to be considered proper peaks. Defaults to 0.5.
        score_measure (int, optional): Intensity measure to use for score calculation. 0: None; 1: Median peak height; 2: Mean peak height; 3: Mean peak area; 4: Total peak area. Defaults to 1.
        n_peaks_influence (int, optional): How the number of peaks influences the score. 0: No influence; 1: Multiplicative, 2: Exponential. Defaults to 1.

    Returns:
        pandas.DataFrame: Spectral data sorted by quality score, with low quality spectra optionally removed.
    """

    start_time = time.perf_counter()

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")

    orig_data = data.copy()

    labels = data.label
    if "file" in data.columns:
        files = data.file
    else:
        files = None

    data = data.loc[:, ~data.columns.isin(["label", "file"])]

    data = limit_range(data, limits[0], limits[1])

    data_bl = baseline_correction(data, method=bl_method)

    peaks = peakRecognition(data_bl, sg_window, threshold)

    scores, intensity_scores, n_peaks = calc_scores(
        data_bl, peaks, score_measure, n_peaks_influence)

    data_sorted = sort_spectra(orig_data, scores)

    data_out = remove_low_quality(data_sorted, n=n, min_n=min_n, min_score=min_score)

    data_out.drop(columns="score", inplace=True)

    end_time = time.perf_counter()

    print(f"Analyzed {len(data)} spectra in {round(end_time-start_time, 2)} seconds.")
    print()
    print(f"Mean Score: {int(np.mean(scores))}")
    print()
    print(f"1st Quartile: {int(np.quantile(scores, 0.25))}")
    print(f"Median Score: {int(np.median(scores))}")
    print(f"3rd Quartile: {int(np.quantile(scores, 0.75))}")
    print()
    print(f"Min Score: {int(np.min(scores))}")
    print(f"Max Score: {int(np.max(scores))}")

    if detailed:
        return data_out, (intensity_scores, n_peaks)
    else:
        return data_out
