import matplotlib.pyplot as plt
import re
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
import seaborn as sns
from math import ceil
from scipy.signal import find_peaks
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc


def plot_spectra_peaks(wns, signal, peaks=None, labels=None, figsize=(8,6)):

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.2)

    line, = ax.plot(wns, signal[0, :])
    if peaks is not None:
        peakmarks = ax.scatter(wns[peaks[0]], signal[0, :][peaks[0]],
                            c="red", marker="x", s=50, zorder=3)
    if labels is not None:
        ax.set_title(labels[0])

    ax.set_xlim(wns[0], wns[-1])
    ax.grid()

    ax.set_xlabel("Raman Shift ($\mathregular{cm^{-1}}$)",
                  fontdict={"weight": "bold", "size": 12})

    class Index:
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(signal)
            ydata = signal[i, :]
            line.set_ydata(ydata)
            if peaks is not None:
                marks = np.array([[wns[peak], signal[i][peak]]
                                for peak in peaks[i]])
                if len(marks) == 0:
                    peakmarks.set_visible(False)
                else:
                    peakmarks.set_visible(True)
                    peakmarks.set_offsets(marks)
            if labels is not None:
                ax.set_title(labels[i])

            ax.relim()
            ax.autoscale_view()
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(signal)
            ydata = signal[i, :]
            line.set_ydata(ydata)
            
            if peaks is not None:
                marks = np.array([[wns[peak], signal[i][peak]]
                                for peak in peaks[i]])
                if len(marks) == 0:
                    peakmarks.set_visible(False)
                else:
                    peakmarks.set_visible(True)
                    peakmarks.set_offsets(marks)
            if labels is not None:
                ax.set_title(labels[i])

            ax.relim()
            ax.autoscale_view()
            plt.draw()

    callback = Index()

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])

    bnext = Button(axnext, "Next")
    bprev = Button(axprev, "Prev")

    bnext.on_clicked(callback.next)
    bprev.on_clicked(callback.prev)

    plt.show()



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


def plot_validation_curve(param, train_scores, test_scores, x_label, y_label, log_scale=False, figsize=(8,6)):
    """Plot the validation curve of a model (train and test scores 
    depending on a parameter value.)

    Args:
        param (list-like): List of parameter values.
        train_scores (np.ndarray): Training scores of the hyperparameter optimization. The last dimension of the array must have the same length as param.
        test_scores (_type_): Test scores of the hyperparameter optimization. The last dimension of the array must have the same length as param.
    """
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)

    test_scores_mean = test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        ax.set_xscale("log")

    ax.plot(param, train_scores_mean, color="k", linestyle="dashed", label="Training")
    ax.plot(param, test_scores_mean, color="k", label="Validation")

    ax.fill_between(param, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, color="k", alpha=0.3)
    ax.fill_between(param, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, color="k", alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label.title())

    ax.grid()
    ax.legend(loc="best")


def plot_val_curves(cv_results, score="accuracy", x_labels=None, y_label="Score", log_scale=False, figsize=(8,6)):
    if not isinstance(cv_results, pd.DataFrame):
        raise TypeError("Pandas DataFrame required.")
    if isinstance(x_labels, str):
        x_labels = [x_labels]

    params = cv_results.filter(regex="param_")
    train_scores = np.asarray(cv_results.filter(regex=f"train_{score}_"))
    test_scores = np.asarray(cv_results.filter(regex=f"test_{score}_"))

    if len(params.columns) == 1:
        
        if x_labels is None:
            x_labels = [params.columns[0]]

        x = np.asarray(params).ravel()
        plot_validation_curve(x, train_scores, test_scores, x_label=x_labels[0], y_label=y_label, log_scale=log_scale, figsize=figsize)
    
    else:
        if x_labels is None:
            x_labels = params.columns
        if isinstance(log_scale, bool):
            log_scale = np.full(len(params.columns), log_scale, dtype=bool)

        max_i = np.unravel_index(np.argmax(test_scores, axis=None), test_scores.shape)[0]
        max_params = params.iloc[max_i,:]

        for i, (param, vals) in enumerate(params.iteritems()):
            max_params_tmp = max_params.drop(param)
            indices = np.ones(len(params), dtype=bool)
            for p, val in max_params_tmp.iteritems():
                indices = np.bitwise_and(indices, params[p] == val)
            x = np.asarray(vals[indices])
            plot_validation_curve(x, train_scores[indices], test_scores[indices], x_label=x_labels[i], y_label=y_label, log_scale=log_scale[i], figsize=figsize)


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


def plot_coefs(coefs, features=None, xlabel=None, ylabel="Coefficient (-)", 
               show_range=False, annotate=False,
               min_height=0, min_dist=None, figsize=(8,6)):

    if isinstance(coefs, pd.DataFrame):
        features = np.asarray(coefs.columns.astype(float))
    elif isinstance(coefs, pd.Series):
        features = np.asarray(coefs.index.astype(float))
    elif features is None:
        features = range(len(coefs[0]))

    coefs = np.asarray(coefs)

    if len(coefs.shape) == 1:
        coefs_plot = coefs
    elif len(coefs.shape) == 2:
        coefs_plot = np.mean(coefs, axis=0)
    else:
        raise ValueError("Only 1 or 2-dimensional arrays are supported.")

    features_0, coefs_neg, coefs_0, coefs_pos = split_by_sign(
        features, coefs_plot)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(c="black", alpha=0.5, linewidth=1)

    ax.plot(features_0, coefs_neg, color="blue")
    ax.plot(features_0, coefs_pos, color="red")
    ax.plot(features_0, coefs_0, color="black")

    if show_range:
        coefs_std = np.std(coefs, axis=0)
        coefs_lower = coefs_plot - coefs_std
        coefs_upper = coefs_plot + coefs_std

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


def plot_confidence_scores(scores, groups, order=None, scale="linear", figsize=(8,6)):

    scores_plot = scores.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)

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


def plot_confusion_matrix(y_pred, y_true, labels, title, figsize=(6,6)):

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    conf_matrices = np.asarray(
        [confusion_matrix(y_true, y_pred[i, :]) for i in range(len(y_pred))]
    )

    conf_matrix_plot = conf_matrices.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    ConfusionMatrixDisplay(conf_matrix_plot).plot(values_format=".1f", ax=ax)

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=16)

    fig.tight_layout()

    plt.plot()


def plot_roc_curve(conf_scores, y, labels, name, figsize=(8,6)):

    if not isinstance(conf_scores, np.ndarray):
        conf_scores = np.asarray(conf_scores)
    mean_fpr = np.linspace(0, 1, 200)
    aucs = []
    tprs = []

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    for row in conf_scores:
        fpr, tpr, _ = roc_curve(y, row)
        ax.plot(fpr, tpr, color="k", alpha=0.2, linewidth=1)
        aucs.append(auc(fpr, tpr))
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1

    aucs_mean = np.mean(aucs)
    aucs_std = np.std(aucs)

    ax.plot(
        mean_fpr, mean_tpr, color="k", linewidth=2,
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

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.legend(loc="lower right")
    fig.tight_layout()

    plt.plot()

    return np.array((mean_fpr, mean_tpr)), np.array((aucs_mean, aucs_std))


def plot_heatmap(combinations, test_scores, x, y, grouping=None, max_cols=None, figsize=(8,6)):

    scores_df = pd.DataFrame(combinations)
    scores_df["test_scores"] = test_scores.mean(axis=0)

    if grouping is None:
        fig, ax = plt.subplots(figsize=figsize)

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


def plot_qc_summary(qc_results, 
                    binrange_peaks=None, 
                    binwidth_peaks=None, 
                    binrange_score=None, 
                    binwidth_score=None,
                    ymax_peaks=None,
                    ymax_score=None, 
                    figsize=(8,6)):

    sns.set(style="ticks")

    fig, ((ax_box1, ax_box2), (ax_hist1, ax_hist2)) = plt.subplots(
        2, 2, sharex="col", gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)

    sns.boxplot(x=qc_results.iloc[:,1], ax=ax_box1)
    sns.boxplot(x=qc_results.iloc[:,0], ax=ax_box2)
    sns.histplot(qc_results.iloc[:,1], ax=ax_hist1, binrange=binrange_peaks, binwidth=binwidth_peaks)
    sns.histplot(qc_results.iloc[:,0], ax=ax_hist2, binrange=binrange_score, binwidth=binwidth_score)

    ax_box1.set(yticks=[])
    ax_box2.set(yticks=[])
    sns.despine(ax=ax_hist1)
    sns.despine(ax=ax_hist2)
    sns.despine(ax=ax_box1, left=True)
    sns.despine(ax=ax_box2, left=True)

    ax_hist1.set_xlabel("Number of Peaks")
    ax_hist2.set_xlabel(qc_results.columns[0])

    ax_hist1.set_ylim([None, ymax_peaks])
    ax_hist2.set_ylim([None, ymax_score])

    ax_box1.tick_params(axis="x", labelbottom=True)
    ax_box2.tick_params(axis="x", labelbottom=True)

    plt.tight_layout()
    plt.show()


def plot_params(params, labels=None, log_scale=False, figsize=(8,6)):
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        params.columns = labels
    
    for param, vals in params.iteritems():
        fig, ax = plt.subplots(figsize=figsize)
        pd.DataFrame(vals).boxplot(ax=ax)
        if log_scale:
            ax.set_yscale("log")
        


def plot_roc_comparison(rocs, aucs, labels=None, regex=None, figsize=(8,6)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    for name, curve in rocs.items():
        if regex:
            if re.search(regex, name):
                ax.plot(curve[0], curve[1], 
                    label=f"{name} (AUC = {aucs[name][0]:.4f} $\pm$ {aucs[name][1]:.4f})")
        else:
            ax.plot(curve[0], curve[1], 
                label=f"{name} (AUC = {aucs[name][0]:.4f} $\pm$ {aucs[name][1]:.4f})")
    
    ax.set_xlabel(
        f"False Positive Rate (Positive label: {labels[1]})",
        fontsize=12
    )

    ax.set_ylabel(
        f"True Positive Rate (Positive label: {labels[1]})",
        fontsize=12
    )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.legend(loc="lower right")
    fig.tight_layout()

    plt.plot()


def boxplot_comparison(data, ylabel, log_scale=False, regex=None, figsize=(8,6)):
    fig, ax = plt.subplots(figsize=figsize)

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if regex:
        data = data.filter(regex=regex)

    sns.boxplot(data=data, 
                ax=ax,
                boxprops={"facecolor": "w"})

    if log_scale:
        ax.semilogy()

    ax.set_ylabel(ylabel)
    ax.grid(axis="y")