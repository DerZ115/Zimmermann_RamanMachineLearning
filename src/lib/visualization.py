import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import ceil
from scipy.signal import find_peaks
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc


def split_by_sign(x, y):
    x1, x2, y1, y2 = np.stack(
        [x[:-1],  x[1:], y[:-1], y[1:]])[:, np.diff(y < 0)]
    xf = x1 + -y1 * (x2 - x1) / (y2 - y1)

    i = np.searchsorted(x, xf)
    x0 = np.insert(x, i, xf)
    y0 = np.insert(y, i, 0)

    y_neg = np.ma.masked_array(y0, mask=y0 > 0)
    y_zero = np.ma.masked_array(y0, mask=y0 != 0)
    y_pos = np.ma.masked_array(y0, mask=y0 < 0)

    return x0, y_neg, y_zero, y_pos


def plot_validation_curve(param, train_scores, test_scores, label, log_scale=False):
    """Plot the validation curve of a model (train and test scores 
    depending on a parameter value.)

    Args:
        param (list-like): List of parameter values.
        train_scores (np.ndarray): Training scores of the hyperparameter optimization. The last dimension of the array must have the same length as param.
        test_scores (_type_): Test scores of the hyperparameter optimization. The last dimension of the array must have the same length as param.
    """
    train_scores_mean = train_scores.mean(axis=0)
    train_scores_std = train_scores.std(axis=0)

    test_scores_mean = test_scores.mean(axis=0)
    test_scores_std = test_scores.std(axis=0)

    fig, ax = plt.subplots()

    if log_scale:
        ax.set_xscale("log")

    ax.plot(param, train_scores_mean, label="Training")
    ax.plot(param, test_scores_mean, label="Validation")

    ax.fill_between(param, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.5)
    ax.fill_between(param, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.5)

    ax.set_xlabel(label)
    ax.set_ylabel("Score")

    ax.grid()
    ax.legend(loc="best")


def annotate_peaks(x, y, min_height=0, min_dist=None):

    if min_dist is None:
        min_dist = len(x) // 100

    peaks_pos = find_peaks(y, height=min_height, distance=min_dist)[0]
    peaks_neg = find_peaks(np.negative(
        y), height=min_height, distance=min_dist)[0]

    for peak in peaks_pos:
        plt.annotate(str(int(x[peak])),
                     (x[peak], y[peak]),
                     xytext=(0, 6),
                     textcoords="offset points",
                     rotation=90,
                     ha="center")

    for peak in peaks_neg:
        plt.annotate(str(int(x[peak])),
                     (x[peak], y[peak]),
                     xytext=(0, -6),
                     textcoords="offset points",
                     rotation=90,
                     ha="center",
                     va="top")


def plot_coefs(features, coefs, method=None, q=0.0,
               xlabel=None, ylabel="Coefficient (-)", show_range=None, annotate=False,
               min_height=0, min_dist=None):

    if method == "mean":
        if q == 0:
            coefs_plot = np.mean(coefs, axis=(0, 1))
        else:
            coefs_low, coefs_high = np.quantile(coefs, (q, 1-q), axis=(0, 1))
            coefs_plot = coefs.mean(axis=(0, 1),
                                    where=(coefs >= coefs_low) &
                                    (coefs <= coefs_high))
    elif method == "median":
        coefs_plot = np.median(coefs, axis=(0, 1))

    else:
        coefs_plot = coefs.ravel()

    if show_range == "std":
        coefs_std = np.std(coefs, axis=(0, 1))
        coefs_lower = coefs_plot - coefs_std
        coefs_upper = coefs_plot + coefs_std
    elif show_range == "iqr":
        coefs_lower, coefs_upper = np.quantile(coefs, (0.25, 0.75))

    features_0, coefs_neg, coefs_0, coefs_pos = split_by_sign(
        features, coefs_plot)

    fig, ax = plt.subplots()
    ax.axhline(c="black", alpha=0.5, linewidth=1)

    ax.plot(features_0, coefs_neg, color="blue")
    ax.plot(features_0, coefs_pos, color="red")
    ax.plot(features_0, coefs_0, color="black")

    if show_range is not None:
        ax.fill_between(features, coefs_lower, coefs_upper,
                        color="grey", alpha=0.7, edgecolor=None)

    if annotate:
        annotate_peaks(features,
                       coefs_plot,
                       min_height=min_height,
                       min_dist=min_dist)

    ax.margins(x=0, y=0.15)
    ax.grid()

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    fig.tight_layout()

    plt.plot()


def plot_confidence_scores(scores, groups, order=None, scale="linear"):

    scores_plot = scores.mean(axis=0)

    fig, ax = plt.subplots()

    if scale == "linear":
        ax.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.7)
    elif scale == "logit":
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.7)

    sns.boxplot(
        x=groups,
        y=scores_plot,
        order=order,
        ax=ax,
        showfliers=False,
        boxprops={"facecolor": "white"}
    )

    sns.stripplot(
        x=groups,
        y=scores_plot,
        order=order,
        ax=ax
    )

    ax.set_yscale(scale)

    if scale == "linear":
        ax.set_ylabel("Confidence score (-)", fontsize=12)
    elif scale == "logit":
        ax.set_ylabel("Probability (-)", fontsize=12)

    fig.tight_layout()

    plt.plot()


def plot_confusion_matrix(y_pred, y_true, labels, subtitle):

    conf_matrices = np.asarray(
        [confusion_matrix(y_true, y_pred[i, :]) for i in range(len(y_pred))]
    )

    conf_matrix_plot = conf_matrices.mean(axis=0)

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay(conf_matrix_plot).plot(values_format=".1f", ax=ax)

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Average Confusion Matrix\n{subtitle}", fontsize=16)

    fig.tight_layout()

    plt.plot()


def plot_roc_curve(conf_scores, y, labels, name):
    aucs = []

    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    for row in conf_scores:
        fpr, tpr, _ = roc_curve(y, row)
        ax.plot(fpr, tpr, color="C0", alpha=0.2, linewidth=1)
        aucs.append(auc(fpr, tpr))

    mean_fpr, mean_tpr, _ = roc_curve(y, conf_scores.mean(axis=0))

    aucs_mean = np.mean(aucs)
    aucs_std = np.std(aucs)

    ax.plot(
        mean_fpr, mean_tpr, color="C0", linewidth=2,
        label=f"{name} (AUC = {aucs_mean:.4f} $\pm$ {aucs_std:.4f})"
    )

    ax.set_xlabel(
        f"False Positive Rate (Positive label: {labels[1]})",
        fontsize=12
    )

    ax.set_ylabel(
        f"True Positive Rate (Positive label: {labels[1]})",
        fontsize=12
    )

    ax.legend(loc="lower right")
    fig.tight_layout()

    plt.plot()

    return np.array((mean_fpr, mean_tpr)), np.array((aucs_mean, aucs_std))


def plot_heatmap(combinations, test_scores, x, y, grouping=None, max_cols=None):

    scores_df = pd.DataFrame(combinations)
    scores_df["test_scores"] = test_scores.mean(axis=0)

    if grouping is None:
        fig, ax = plt.subplots()

        piv = pd.pivot(scores_df, index=y, columns=x, values="test_scores")

        sns.heatmap(piv,
                    vmin=np.min(scores_df.test_scores),
                    vmax=np.max(scores_df.test_scores),
                    cmap="viridis",
                    annot=True,
                    cbar=False,
                    square=True,
                    ax=ax)

    else:

        groups = scores_df.groupby(grouping)

        if max_cols is None:
            if isinstance(grouping, str):
                ncols = len(groups)
            else:
                ncols = len(scores_df[grouping[0]].unique)
        else:
            if len(groups) < max_cols:
                n_cols = len(groups)
            else:
                n_cols = max_cols

        nrows = ceil(len(groups) / ncols)

        fig, axes = plt.subplots(nrows, ncols)

        for (title, group), ax in zip(groups, np.ravel(axes)):
            piv = pd.pivot(group, index=y, columns=x, values="test_scores")
            sns.heatmap(piv,
                        vmin=np.min(scores_df.test_scores),
                        vmax=np.max(scores_df.test_scores),
                        cmap="viridis",
                        annot=True,
                        square=True,
                        ax=ax,
                        cbar=False)
            ax.set_title(title)
