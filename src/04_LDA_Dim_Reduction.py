import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import NMF, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from lib.crossvalidation import randomized_cv
from lib.misc import load_data
from lib.preprocessing import PeakPicker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single csv file from individual Raman spectra."
    )

    parser.add_argument("-f", "--file", metavar="PATH", type=str, nargs=1, action="store",
                        help=".csv file containing the spectral data.", required=True)
    parser.add_argument("-o", "--out", metavar="PATH", type=str, nargs=1, action="store",
                        help="Path for the output directory.", required=True)
    parser.add_argument("-l", "--limits", metavar=("LOW", "HIGH"), type=float, nargs=2, action="store",
                        help="Limits for reducing the spectral range.", required=False, default=(None, None))

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()

    path_in = Path(args.file[0])
    path_out = Path(args.out[0])

    filename = path_in.stem
    data = load_data(path_in)

    X = data.loc[:, data.columns != "label"]
    wns = np.asarray(X.columns.astype(float))
    X = np.asarray(X)
    y = np.asarray(data.label)
    y, y_key = pd.factorize(y)

    scoring = [
    "accuracy", 
    "f1",
    "roc_auc"
    ]

    ### LDA only

    clf = LinearDiscriminantAnalysis()

    clf, ct_results, cv_result = randomized_cv(clf, X, y, ct_scoring=scoring)

    




