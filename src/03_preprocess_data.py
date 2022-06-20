import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer

from lib.misc import load_data
from lib.preprocessing import BaselineCorrector, RangeLimiter, SavGolFilter


def preprocess(data):
    X = data.loc[:, data.columns != "label"]
    wns = np.asarray(X.columns.astype(float))
    X = np.asarray(X)
    y = np.asarray(data.label)

    X = BaselineCorrector().fit_transform(X)

    rl = RangeLimiter(lim=(450, 1670), reference=wns)
    X = rl.fit_transform(X)
    wns_reduced = wns[rl.lim_[0]:rl.lim_[1]]

    X = SavGolFilter().fit_transform(X)

    X = Normalizer().fit_transform(X)

    data_prep = pd.DataFrame(X, columns=wns_reduced)
    data_prep.insert(0, "label", y)

    return data_prep

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single csv file from individual Raman spectra."
    )

    parser.add_argument("-f", "--file", metavar="PATH", type=str, nargs=1, action="store",
                        help=".csv file containing the spectral data.", required=True)
    parser.add_argument("-o", "--out", metavar="PATH", type=str, nargs=1, action="store",
                        help="Path for the output file.", required=True)
    parser.add_argument("-l", "--limits", metavar=("LOW", "HIGH"), type=float, nargs=2, action="store",
                        help="Limits for reducing the spectral range.", required=False, default=(None, None))

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()

    path_in = Path(args.file[0])
    path_out = Path(args.out[0])

    filename = path_in.stem

    if path_out.is_dir:
        path_out = path_out / (filename + "_preprocessed.csv")

    data = load_data(path_in)

    data_prep = preprocess(data)

    data_prep.to_csv(path_out, index=False)
